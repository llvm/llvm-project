export {};

/// The symbol type we get from the lldb-dap server
export declare interface SymbolType {
  id: number;
  isDebug: boolean;
  isSynthetic: boolean;
  isExternal: boolean;
  type: string;
  fileAddress: number;
  loadAddress?: number;
  size: number;
  name: string;
}
