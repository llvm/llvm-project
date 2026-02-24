//===-- DAPTypes.h ---------------------------------------------------===//
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

#include "Protocol/ProtocolBase.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>

namespace lldb_dap::protocol {

enum ReferenceKind : uint8_t {
  eReferenceKindTemporary = 0,
  eReferenceKindPermanent = 1,
  eReferenceKindScope = 1 << 1,
  eReferenceKindInvalid = 0xFF
};

/// The var_ref_t hold two values, the `ReferenceKind` and the
/// `variablesReference`.
struct var_ref_t {
private:
  static constexpr uint32_t k_kind_bit_size = sizeof(ReferenceKind) * 8;
  static constexpr uint32_t k_reference_bit_size =
      std::numeric_limits<uint32_t>::digits - k_kind_bit_size;
  static constexpr uint32_t k_reference_bit_mask =
      (1 << k_reference_bit_size) - 1;
  static constexpr uint32_t k_kind_mask = 0xFF;

public:
  static constexpr uint32_t k_invalid_var_ref = UINT32_MAX;
  static constexpr uint32_t k_no_child = 0;

  explicit constexpr var_ref_t(uint32_t reference, ReferenceKind kind)
      : reference(reference), kind(kind) {}

  explicit constexpr var_ref_t(uint32_t masked_ref = k_invalid_var_ref)
      : reference(masked_ref & k_reference_bit_mask),
        kind((masked_ref >> k_reference_bit_size) & k_kind_mask) {}

  [[nodiscard]] constexpr uint32_t AsUInt32() const {
    return (kind << k_reference_bit_size) | reference;
  };

  [[nodiscard]] constexpr ReferenceKind Kind() const {
    const auto current_kind = static_cast<ReferenceKind>(kind);
    switch (current_kind) {
    case eReferenceKindTemporary:
    case eReferenceKindPermanent:
    case eReferenceKindScope:
      return current_kind;
    default:
      return eReferenceKindInvalid;
    }
  }

  [[nodiscard]] constexpr uint32_t Reference() const { return reference; }

  // We should be able to store at least 8 million variables for each store
  // type at every stopped state.
  static constexpr uint32_t k_variables_reference_threshold = 8'000'000;
  static constexpr uint32_t k_max_variables_references =
      k_reference_bit_mask - 1;
  static_assert((k_max_variables_references >
                 k_variables_reference_threshold) &&
                "not enough variablesReferences to store 8 million variables.");

private:
  uint32_t reference : k_reference_bit_size;
  uint32_t kind : k_kind_bit_size;
};
static_assert(sizeof(var_ref_t) == sizeof(uint32_t) &&
              "the size of var_ref_t must be equal to the size of uint32_t.");

bool fromJSON(const llvm::json::Value &, var_ref_t &, llvm::json::Path);
inline llvm::json::Value toJSON(const var_ref_t &var_ref) {
  return var_ref.AsUInt32();
}

/// Data used to help lldb-dap resolve breakpoints persistently across different
/// sessions. This information is especially useful for assembly breakpoints,
/// because `sourceReference` can change across sessions. For regular source
/// breakpoints the path and line are the same For each session.
struct PersistenceData {
  /// The source module path.
  String module_path;

  /// The symbol name of the Source.
  String symbol_name;
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
  uint32_t id = 0;

  /// True if this symbol is debug information in a symbol.
  bool isDebug = false;

  /// True if this symbol is not actually in the symbol table, but synthesized
  /// from other info in the object file.
  bool isSynthetic = false;

  /// True if this symbol is globally visible.
  bool isExternal = false;

  /// The symbol type.
  lldb::SymbolType type = lldb::eSymbolTypeInvalid;

  /// The symbol file address.
  lldb::addr_t fileAddress = LLDB_INVALID_ADDRESS;

  /// The symbol load address.
  std::optional<lldb::addr_t> loadAddress;

  /// The symbol size.
  lldb::addr_t size = 0;

  /// The symbol name.
  String name;
};
bool fromJSON(const llvm::json::Value &, Symbol &, llvm::json::Path);
llvm::json::Value toJSON(const Symbol &);

} // namespace lldb_dap::protocol

#endif
