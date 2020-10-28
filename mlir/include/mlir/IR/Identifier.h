//===- Identifier.h - MLIR Identifier Class ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_IDENTIFIER_H
#define MLIR_IR_IDENTIFIER_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringMapEntry.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
class MLIRContext;

/// This class represents a uniqued string owned by an MLIRContext.  Strings
/// represented by this type cannot contain nul characters, and may not have a
/// zero length.
///
/// This is a POD type with pointer size, so it should be passed around by
/// value.  The underlying data is owned by MLIRContext and is thus immortal for
/// almost all clients.
class Identifier {
  using EntryType = llvm::StringMapEntry<llvm::NoneType>;

public:
  /// Return an identifier for the specified string.
  static Identifier get(StringRef str, MLIRContext *context);
  Identifier(const Identifier &) = default;
  Identifier &operator=(const Identifier &other) = default;

  /// Return a StringRef for the string.
  StringRef strref() const { return entry->first(); }

  /// Identifiers implicitly convert to StringRefs.
  operator StringRef() const { return strref(); }

  /// Return an std::string.
  std::string str() const { return strref().str(); }

  /// Return a null terminated C string.
  const char *c_str() const { return entry->getKeyData(); }

  /// Return a pointer to the start of the string data.
  const char *data() const { return entry->getKeyData(); }

  /// Return the number of bytes in this string.
  unsigned size() const { return entry->getKeyLength(); }

  const char *begin() const { return data(); }
  const char *end() const { return entry->getKeyData() + size(); }

  bool operator==(Identifier other) const { return entry == other.entry; }
  bool operator!=(Identifier rhs) const { return !(*this == rhs); }

  void print(raw_ostream &os) const;
  void dump() const;

  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(entry);
  }
  static Identifier getFromOpaquePointer(const void *entry) {
    return Identifier(static_cast<const EntryType *>(entry));
  }

  /// Compare the underlying StringRef.
  int compare(Identifier rhs) const { return strref().compare(rhs.strref()); }

private:
  /// This contains the bytes of the string, which is guaranteed to be nul
  /// terminated.
  const EntryType *entry;
  explicit Identifier(const EntryType *entry) : entry(entry) {}
};

inline raw_ostream &operator<<(raw_ostream &os, Identifier identifier) {
  identifier.print(os);
  return os;
}

// Identifier/Identifier equality comparisons are defined inline.
inline bool operator==(Identifier lhs, StringRef rhs) {
  return lhs.strref() == rhs;
}
inline bool operator!=(Identifier lhs, StringRef rhs) { return !(lhs == rhs); }

inline bool operator==(StringRef lhs, Identifier rhs) {
  return rhs.strref() == lhs;
}
inline bool operator!=(StringRef lhs, Identifier rhs) { return !(lhs == rhs); }

// Make identifiers hashable.
inline llvm::hash_code hash_value(Identifier arg) {
  // Identifiers are uniqued, so we can just hash the pointer they contain.
  return llvm::hash_value(arg.getAsOpaquePointer());
}
} // end namespace mlir

namespace llvm {
// Identifiers hash just like pointers, there is no need to hash the bytes.
template <>
struct DenseMapInfo<mlir::Identifier> {
  static mlir::Identifier getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<const void *>::getEmptyKey();
    return mlir::Identifier::getFromOpaquePointer(pointer);
  }
  static mlir::Identifier getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<const void *>::getTombstoneKey();
    return mlir::Identifier::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::Identifier val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Identifier lhs, mlir::Identifier rhs) {
    return lhs == rhs;
  }
};

/// The pointer inside of an identifier comes from a StringMap, so its alignment
/// is always at least 4 and probably 8 (on 64-bit machines).  Allow LLVM to
/// steal the low bits.
template <>
struct PointerLikeTypeTraits<mlir::Identifier> {
public:
  static inline void *getAsVoidPointer(mlir::Identifier i) {
    return const_cast<void *>(i.getAsOpaquePointer());
  }
  static inline mlir::Identifier getFromVoidPointer(void *p) {
    return mlir::Identifier::getFromOpaquePointer(p);
  }
  static constexpr int NumLowBitsAvailable = 2;
};

} // end namespace llvm
#endif
