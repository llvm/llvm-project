import type { CellComponent, ColumnDefinition } from "tabulator-tables";
import type { DAPSymbolType } from ".."

function get_tabulator_hexa_formatter(padding: number): (cell: CellComponent) => string {
  return (cell: CellComponent) => {
    const val = cell.getValue();
    if (val === undefined || val === null) {
      return "";
    }

    return val !== undefined ? "0x" + val.toString(16).toLowerCase().padStart(padding, "0") : "";
  };
}

const SYMBOL_TABLE_COLUMNS: ColumnDefinition[] = [
  { title: "User ID", field: "userId", headerTooltip: true, sorter: "number", widthGrow: 0.8 },
  {
    title: "Name",
    field: "name",
    headerTooltip: true,
    sorter: "string",
    widthGrow: 2.5,
    minWidth: 200,
    tooltip : (_event: MouseEvent, cell: CellComponent) => {
      const rowData = cell.getRow().getData();
      return rowData.name;
    }
  },
  {
    title: "Debug",
    field: "isDebug",
    headerTooltip: true,
    hozAlign: "center",
    widthGrow: 0.8,
    formatter: "tickCross",
    formatterParams: {
      tickElement: "✔",
      crossElement: false,
    }
  },
  {
    title: "Synthetic",
    field: "isSynthetic",
    headerTooltip: true,
    hozAlign: "center",
    widthGrow: 0.8,
    formatter: "tickCross",
    formatterParams: {
      tickElement: "✔",
      crossElement: false,
    }
  },
  {
    title: "External",
    field: "isExternal",
    headerTooltip: true,
    hozAlign: "center",
    widthGrow: 0.8,
    formatter: "tickCross",
    formatterParams: {
      tickElement: "✔",
      crossElement: false,
    }
  },
  { title: "Type", field: "type", sorter: "string" },
  {
    title: "File Address",
    field: "fileAddress",
    headerTooltip: true,
    sorter: "number",
    widthGrow : 1.25,
    formatter: get_tabulator_hexa_formatter(16),
  },
  {
    title: "Load Address",
    field: "loadAddress",
    headerTooltip: true,
    sorter: "number",
    widthGrow : 1.25,
    formatter: get_tabulator_hexa_formatter(16),
  },
  { title: "Size", field: "size", headerTooltip: true, sorter: "number", formatter: get_tabulator_hexa_formatter(8) },
];

const vscode = acquireVsCodeApi();
const previousState: any = vscode.getState();

declare const Tabulator: any; // HACK: real definition comes from tabulator.min.js
const SYMBOLS_TABLE = new Tabulator("#symbols-table", {
  height: "100vh",
  columns: SYMBOL_TABLE_COLUMNS,
  layout: "fitColumns",
  data: previousState?.symbols || [],
});

function updateSymbolsTable(symbols: DAPSymbolType[]) {
  SYMBOLS_TABLE.setData(symbols);
}

window.addEventListener("message", (event: MessageEvent<any>) => {
  const message = event.data;
  switch (message.command) {
    case "updateSymbols":
      vscode.setState({ symbols: message.symbols });
      updateSymbolsTable(message.symbols);
      break;
  }
});

