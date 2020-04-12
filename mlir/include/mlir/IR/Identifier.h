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
#include "llvm/ADT/StringRef.h"
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
public:
  /// Return an identifier for the specified string.
  static Identifier get(StringRef str, MLIRContext *context);
  Identifier(const Identifier &) = default;
  Identifier &operator=(const Identifier &other) = default;

  /// Return a StringRef for the string.
  StringRef strref() const { return StringRef(pointer, size()); }

  /// Identifiers implicitly convert to StringRefs.
  operator StringRef() const { return strref(); }

  /// Return an std::string.
  std::string str() const { return strref().str(); }

  /// Return a null terminated C string.
  const char *c_str() const { return pointer; }

  /// Return a pointer to the start of the string data.
  const char *data() const { return pointer; }

  /// Return the number of bytes in this string.
  unsigned size() const { return ::strlen(pointer); }

  /// Return true if this identifier is the specified string.
  bool is(StringRef string) const {
    // Note: this can't use memcmp, because memcmp doesn't guarantee that it
    // will stop reading both buffers if one is shorter than the other.
    return strncmp(pointer, string.data(), string.size()) == 0 &&
           pointer[string.size()] == '\0';
  }

  const char *begin() const { return pointer; }
  const char *end() const { return pointer + size(); }

  void print(raw_ostream &os) const;
  void dump() const;

  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(pointer);
  }
  static Identifier getFromOpaquePointer(const void *pointer) {
    return Identifier((const char *)pointer);
  }

private:
  /// These are the bytes of the string, which is a nul terminated string.
  const char *pointer;
  explicit Identifier(const char *pointer) : pointer(pointer) {}
};

inline raw_ostream &operator<<(raw_ostream &os, Identifier identifier) {
  identifier.print(os);
  return os;
}

inline bool operator==(Identifier lhs, Identifier rhs) {
  return lhs.data() == rhs.data();
}

inline bool operator!=(Identifier lhs, Identifier rhs) {
  return lhs.data() != rhs.data();
}

inline bool operator==(Identifier lhs, StringRef rhs) { return lhs.is(rhs); }
inline bool operator!=(Identifier lhs, StringRef rhs) { return !lhs.is(rhs); }
inline bool operator==(StringRef lhs, Identifier rhs) { return rhs.is(lhs); }
inline bool operator!=(StringRef lhs, Identifier rhs) { return !rhs.is(lhs); }

// Make identifiers hashable.
inline llvm::hash_code hash_value(Identifier arg) {
  // Identifiers are uniqued, so we can just hash the pointer they contain.
  return llvm::hash_value(static_cast<const void *>(arg.data()));
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
    return DenseMapInfo<const void *>::getHashValue(val.data());
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
