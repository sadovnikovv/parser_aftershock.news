# Aftershock compact export (`export_compact.py`) 

## Назначение 
Скрипт читает исходные SQLite-базы с таблицей `articles` (по маске `aftershock_articles_*.sqlite`) и делает “компактную” базу `articles_compact` + экспорт в Excel (и опционально CSV).   
Цель — убрать тяжёлые поля (полный текст, summary, sha1 и т.п.) и получить удобный для анализа датасет (nid/дата/направление/заголовок/теги). 

## Вход/выход 
Входные файлы задаются `INPUT_GLOB = "aftershock_articles_*.sqlite"`, а результаты кладутся в папку `OUT_DIR = "compact_out"`.   
Выходные файлы именуются с номером части и timestamp, например `aftershock_compact_001_YYYY-MM-DD_HH-MM-SS.sqlite` и `aftershock_compact_001_YYYY-MM-DD_HH-MM-SS.xlsx`. 

## Что именно экспортируется 
В компактную таблицу `articles_compact` пишутся колонки: `nid`, `published_day`, `direction`, `direction_confidence`, `title`, `tags`, `tags_json`.   
`published_day` вычисляется как первые 10 символов `published_at` (формат `YYYY-MM-DD`), а если `published_at` пуст, то делается попытка распарсить `published_raw` через `dateparser` (если он установлен).   
Поле `tags` — это строка, собранная из `tags_json` (JSON-список), а `title` обрезается до `MAX_TITLE_CHARS`. 

## Ротация и ограничения размера 
Если текущая compact-БД превышает `MAX_BYTES` (по умолчанию 35 MB), скрипт делает `VACUUM`, и если размер всё равно большой — ротирует на следующую “часть”.   
Excel-файл тоже может “раздуваться”, поэтому есть лимит строк на файл `ROWS_PER_XLSX_PART` (по умолчанию 150_000). 

## Экспорт в Excel/CSV 
В Excel выводится подмножество колонок без `tags_json`: `nid`, `published_day`, `direction`, `direction_confidence`, `title`, `tags`.   
Экспорт идёт чанками через `pandas.read_sql_query(..., chunksize=...)`, а опционально включается авто-подбор ширины колонок (`AUTO_FIT_COLUMNS`). 
