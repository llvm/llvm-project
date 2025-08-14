import type { CellComponent, ColumnDefinition } from "tabulator-tables";

const TABULATOR_HEXA_FORMATTER = (cell: CellComponent) => {
    const val = cell.getValue();
    return val !== undefined ? "0x" + val.toString(16).toUpperCase() : "";
};

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
        formatter: TABULATOR_HEXA_FORMATTER,
    },
    {
        title: "Load Address",
        field: "loadAddress",
        sorter: "number",
        formatter: TABULATOR_HEXA_FORMATTER,
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

declare const Tabulator: any; // HACK: real definition comes from tabulator.min.js
const SYMBOLS_TABLE = new Tabulator("#table", {
    columns: SYMBOL_TABLE_COLUMNS,
    layout: "fitData",
    data: [],
});

window.addEventListener("message", (event: MessageEvent<any>) => {
    console.log("FUCK1");
    const message = event.data;
    switch (message.command) {
        case "updateSymbols":
            console.log("Received symbols update:", message.symbols);
            SYMBOLS_TABLE.setData(message.symbols);
            break;
    }
});

