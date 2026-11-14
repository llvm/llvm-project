/**
 * @fileoverview A re-exported Debug Adapter Protocol types
 * with lldb-dap's extensions.
 * This allows using the types directly without manually extending
 * the DAP type and adding an LLDB prefix.
 * The extensions are documented in lldb/docs/resources/lldbdap-extensions.md.
 */

import type { DebugProtocol } from "@vscode/debugprotocol";

export type { DebugProtocol as dap };

declare module "@vscode/debugprotocol" {
  namespace DebugProtocol {
    interface Capabilities {
      supportsModuleSymbolsRequest?: boolean;
    }

    interface StackFrame {
      $__lldb_isOptimized?: boolean;
    }

    interface ModuleSymbolsArguments {
      /** The UUID of the module. */
      moduleId: string;
      moduleName: string;
      /** The index of the first symbol to return. If omitted, starts at 0. */
      startIndex?: number;
      /**
       * The number of symbols to return. If count is missing or 0, all symbols
       * are returned.
       */
      count?: number;
    }

    interface ModuleSymbol {
      id: number;
      name: string;
      /** The size of the symbol in bytes. */
      size: number;
      isDebug: boolean;
      /** The symbol was synthesized by the debugger. */
      isSynthetic: boolean;
      isExternal: boolean;
      /** The `lldb::SymbolType`, e.g. "Code", "Trampoline", "Data". */
      type: string;
      /** The address of the symbol in the module file. */
      fileAddress: number;
      /**
       * The address of the symbol in the process. `null` if the module is not
       * loaded.
       */
      loadAddress?: number | null;
    }
  }
}
