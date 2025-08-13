import type { ColumnDefinition } from "tabulator-tables";

const SYMBOL_TABLE_COLUMNS: ColumnDefinition[] = [
    { title: "User ID", field: "userId", sorter: "number" },
    { title: "Debug", field: "isDebug", sorter: "boolean" },
    { title: "Synthetic", field: "isSynthetic", sorter: "boolean" },
    { title: "External", field: "isExternal", sorter: "boolean" },
    { title: "Type", field: "type", sorter: "string" },
    {
        title: "File Address",
        field: "fileAddress",
        sorter: "number",
        formatter: cell => "0x" + cell.getValue().toString(16).toUpperCase()
    },
    {
        title: "Load Address",
        field: "loadAddress",
        sorter: "number",
        formatter: cell => {
            const val = cell.getValue();
            return val !== undefined ? "0x" + val.toString(16).toUpperCase() : "";
        }
    },
    { title: "Size", field: "size", sorter: "number" },
    {
        title: "Name",
        field: "name",
        sorter: "string",
        widthGrow: 2,
        minWidth: 200
    }
]

console.log("FUCK");
console.log("Symbols table columns:", SYMBOL_TABLE_COLUMNS);

declare const Tabulator: any; // HACK: real definition comes from tabulator.min.js
const SYMBOLS_TABLE = new Tabulator("#symbols-table", {
    columns: SYMBOL_TABLE_COLUMNS,
    layout: "fitData",
    data: [],
});

window.addEventListener("message", (event: MessageEvent<any>) => {
    const message = event.data;
    switch (message.type) {
        case "updateSymbols":
            SYMBOLS_TABLE.setData(message.symbols);
            break;
    }
});

