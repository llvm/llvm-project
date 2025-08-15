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
  { title: "User ID", field: "userId", sorter: "number", widthGrow: 0.8 },
  {
    title: "Name",
    field: "name",
    sorter: "string",
    widthGrow: 3,
    minWidth: 200,
    tooltip : (_event: MouseEvent, cell: CellComponent) => {
      const rowData = cell.getRow().getData();
      return rowData.name;
    }
  },
  {
    title: "DSX",
    hozAlign: "center",
    widthGrow: 0.8,
    headerTooltip : "Debug / Synthetic / External",
    formatter: (cell: CellComponent) => {
      const rowData = cell.getRow().getData();
      let label = "";
      label += rowData.isDebug ? "D" : "";
      label += rowData.isSynthetic ? "S" : "";
      label += rowData.isExternal ? "X" : "";
      return label;
    },
    sorter: (_a, _b, aRow, bRow) => {
      const valuesA = [aRow.getData().isDebug, aRow.getData().isSynthetic, aRow.getData().isExternal];
      const valuesB = [bRow.getData().isDebug, bRow.getData().isSynthetic, bRow.getData().isExternal];

      return valuesA < valuesB ? -1 : valuesA > valuesB ? 1 : 0;
    }
  },
  { title: "Type", field: "type", sorter: "string" },
  {
    title: "File Address",
    field: "fileAddress",
    sorter: "number",
    widthGrow : 1.25,
    formatter: get_tabulator_hexa_formatter(16),
  },
  {
    title: "Load Address",
    field: "loadAddress",
    sorter: "number",
    widthGrow : 1.25,
    formatter: get_tabulator_hexa_formatter(16),
  },
  { title: "Size", field: "size", sorter: "number", formatter: get_tabulator_hexa_formatter(8) },
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

