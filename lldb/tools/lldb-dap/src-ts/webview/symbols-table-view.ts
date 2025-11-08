import type { CellComponent, ColumnDefinition } from "tabulator-tables";
import type { SymbolType } from ".."

/// SVG from https://github.com/olifolkerd/tabulator/blob/master/src/js/modules/Format/defaults/formatters/tickCross.js
/// but with the default font color.
/// hopefully in the future we can set the color as parameter: https://github.com/olifolkerd/tabulator/pull/4791
const TICK_ELEMENT = `<svg enable-background="new 0 0 24 24" height="14" width="14" viewBox="0 0 24 24" xml:space="preserve" ><path fill="var(--vscode-editor-foreground)" clip-rule="evenodd" d="M21.652,3.211c-0.293-0.295-0.77-0.295-1.061,0L9.41,14.34  c-0.293,0.297-0.771,0.297-1.062,0L3.449,9.351C3.304,9.203,3.114,9.13,2.923,9.129C2.73,9.128,2.534,9.201,2.387,9.351  l-2.165,1.946C0.078,11.445,0,11.63,0,11.823c0,0.194,0.078,0.397,0.223,0.544l4.94,5.184c0.292,0.296,0.771,0.776,1.062,1.07  l2.124,2.141c0.292,0.293,0.769,0.293,1.062,0l14.366-14.34c0.293-0.294,0.293-0.777,0-1.071L21.652,3.211z" fill-rule="evenodd"/></svg>`;

function getTabulatorHexaFormatter(padding: number): (cell: CellComponent) => string {
  return (cell: CellComponent) => {
    const val = cell.getValue();
    if (val === undefined || val === null) {
      return "";
    }

    return val !== undefined ? "0x" + val.toString(16).toLowerCase().padStart(padding, "0") : "";
  };
}

const SYMBOL_TABLE_COLUMNS: ColumnDefinition[] = [
  { title: "ID", field: "id", headerTooltip: true, sorter: "number", widthGrow: 0.6 },
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
      tickElement: TICK_ELEMENT,
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
      tickElement: TICK_ELEMENT,
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
      tickElement: TICK_ELEMENT,
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
    formatter: getTabulatorHexaFormatter(16),
  },
  {
    title: "Load Address",
    field: "loadAddress",
    headerTooltip: true,
    sorter: "number",
    widthGrow : 1.25,
    formatter: getTabulatorHexaFormatter(16),
  },
  { title: "Size", field: "size", headerTooltip: true, sorter: "number", formatter: getTabulatorHexaFormatter(8) },
];

const vscode = acquireVsCodeApi();
const previousState: any = vscode.getState();

declare const Tabulator: any; // HACK: real definition comes from tabulator.min.js
const SYMBOLS_TABLE = new Tabulator("#symbols-table", {
  height: "100vh",
  columns: SYMBOL_TABLE_COLUMNS,
  layout: "fitColumns",
  selectableRows: false,
  data: previousState?.symbols || [],
});

function updateSymbolsTable(symbols: SymbolType[]) {
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

