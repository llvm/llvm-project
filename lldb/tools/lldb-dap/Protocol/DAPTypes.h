//===-- ProtocolTypes.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains private DAP types used in the protocol.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_PROTOCOL_DAP_TYPES_H
#define LLDB_TOOLS_LLDB_DAP_PROTOCOL_DAP_TYPES_H

#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <string>

namespace lldb_dap::protocol {

/// Data used to help lldb-dap resolve breakpoints persistently across different
/// sessions. This information is especially useful for assembly breakpoints,
/// because `sourceReference` can change across sessions. For regular source
/// breakpoints the path and line are the same For each session.
struct PersistenceData {
  /// The source module path.
  std::string module_path;

  /// The symbol name of the Source.
  std::string symbol_name;
};
bool fromJSON(const llvm::json::Value &, PersistenceData &, llvm::json::Path);
llvm::json::Value toJSON(const PersistenceData &);

/// Custom source data used by lldb-dap.
/// This data should help lldb-dap identify sources correctly across different
/// sessions.
struct SourceLLDBData {
  /// Data that helps lldb resolve this source persistently across different
  /// sessions.
  std::optional<PersistenceData> persistenceData;
};
bool fromJSON(const llvm::json::Value &, SourceLLDBData &, llvm::json::Path);
llvm::json::Value toJSON(const SourceLLDBData &);

struct Symbol {
  /// The symbol id, usually the original symbol table index.
  uint32_t id;

  /// True if this symbol is debug information in a symbol.
  bool isDebug;

  /// True if this symbol is not actually in the symbol table, but synthesized
  /// from other info in the object file.
  bool isSynthetic;

  /// True if this symbol is globally visible.
  bool isExternal;

  /// The symbol type.
  lldb::SymbolType type;

  /// The symbol file address.
  lldb::addr_t fileAddress;

  /// The symbol load address.
  std::optional<lldb::addr_t> loadAddress;

  /// The symbol size.
  lldb::addr_t size;

  /// The symbol name.
  std::string name;
};
bool fromJSON(const llvm::json::Value &, Symbol &, llvm::json::Path);
llvm::json::Value toJSON(const Symbol &);

} // namespace lldb_dap::protocol

#endif
