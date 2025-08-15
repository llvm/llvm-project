export {};

/// The symbol type we get from the lldb-dap server
export type DAPSymbolType = {
  userId: number;
  isDebug: boolean;
  isSynthetic: boolean;
  isExternal: boolean;
  type: string;
  fileAddress: number;
  loadAddress?: number;
  size: number;
  name: string;
};
