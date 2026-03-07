//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Mutable in-memory representation of BTF type information.
///
/// BTFBuilder provides an interface for constructing and merging .BTF
/// sections. Types and strings can be added individually or merged from
/// raw .BTF section data parsed from ELF object files. The result can
/// be serialized back to binary .BTF format.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_BTF_BTFBUILDER_H
#define LLVM_DEBUGINFO_BTF_BTFBUILDER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/BTF/BTF.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

namespace llvm {

/// A mutable container for BTF type information that supports construction,
/// merging, and serialization.
///
/// Types are stored as contiguous raw bytes in native byte order.
/// Type IDs are 1-based (ID 0 = void, never stored).
class BTFBuilder {
  // String table: concatenated NUL-terminated strings.
  // Offset 0 is always the empty string (single NUL byte).
  SmallVector<char, 0> Strings;

  // Raw type data in native byte order. Types are stored sequentially,
  // each as CommonType header followed by kind-specific tail data.
  SmallVector<uint8_t, 0> TypeData;

  // TypeOffsets[i] is the byte offset in TypeData for type ID (i+1).
  SmallVector<uint32_t, 0> TypeOffsets;

public:
  LLVM_ABI BTFBuilder();

  /// Add a string, returning its offset in the string table.
  LLVM_ABI uint32_t addString(StringRef S);

  /// Add a type header, returning the new 1-based type ID.
  /// Append kind-specific tail data with addTail() immediately after.
  LLVM_ABI uint32_t addType(const BTF::CommonType &Header);

  /// Append kind-specific tail data for the most recently added type.
  template <typename T> void addTail(const T &Data) {
    const auto *Ptr = reinterpret_cast<const uint8_t *>(&Data);
    TypeData.append(Ptr, Ptr + sizeof(Data));
  }

  /// Merge all types and strings from a raw .BTF section, remapping
  /// type IDs and string offsets. Returns the first new type ID.
  LLVM_ABI Expected<uint32_t> merge(StringRef RawBTFSection,
                                    bool IsLittleEndian);

  /// Look up a type by 1-based ID. Returns nullptr for invalid IDs.
  LLVM_ABI const BTF::CommonType *findType(uint32_t Id) const;

  /// Get raw bytes for a type entry (CommonType + tail data).
  LLVM_ABI ArrayRef<uint8_t> getTypeBytes(uint32_t Id) const;

  /// Get mutable raw bytes for a type entry.
  LLVM_ABI MutableArrayRef<uint8_t> getMutableTypeBytes(uint32_t Id);

  /// Look up a string by offset in the string table.
  LLVM_ABI StringRef findString(uint32_t Offset) const;

  /// Number of types, excluding void (type 0).
  uint32_t typesCount() const { return TypeOffsets.size(); }

  /// Compute the byte size of a type entry from its CommonType header.
  LLVM_ABI static size_t typeByteSize(const BTF::CommonType *T);

  /// Returns true if CommonType.Type is a type reference for this kind.
  LLVM_ABI static bool hasTypeRef(uint32_t Kind);

  /// Serialize to binary .BTF format, appending to Out.
  LLVM_ABI void write(SmallVectorImpl<uint8_t> &Out,
                      bool IsLittleEndian) const;
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_BTF_BTFBUILDER_H
