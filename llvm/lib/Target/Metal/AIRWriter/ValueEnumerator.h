//===- ValueEnumerator.h - AIR value/type/metadata enumerator ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_VALUEENUMERATOR_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_VALUEENUMERATOR_H

#include "PointeeTypeMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include <vector>

namespace llvm {
namespace metal {

/// Type table entry: LLVM Type* + pointee (for pointer types).
/// Non-pointer types have pointee = nullptr.
/// This allows multiple typed pointer entries for the same LLVM ptr type
/// (e.g., float addrspace(3)* AND [128 x float] addrspace(3)*).
struct TypeEntry {
  llvm::Type *type;
  llvm::Type *pointee; // nullptr for non-pointers

  bool operator==(const TypeEntry &O) const {
    return type == O.type && pointee == O.pointee;
  }
};

} // namespace metal

// DenseMap support for TypeEntry
template <> struct DenseMapInfo<metal::TypeEntry> {
  static metal::TypeEntry getEmptyKey() {
    return {DenseMapInfo<Type *>::getEmptyKey(), nullptr};
  }
  static metal::TypeEntry getTombstoneKey() {
    return {DenseMapInfo<Type *>::getTombstoneKey(), nullptr};
  }
  static unsigned getHashValue(const metal::TypeEntry &E) {
    return hash_combine(DenseMapInfo<Type *>::getHashValue(E.type),
                        DenseMapInfo<Type *>::getHashValue(E.pointee));
  }
  static bool isEqual(const metal::TypeEntry &A, const metal::TypeEntry &B) {
    return A == B;
  }
};

namespace metal {

class ValueEnumerator {
public:
  // Type table: each entry is (Type*, pointee) for correct typed pointer
  // emission
  std::vector<TypeEntry> types;
  llvm::DenseMap<TypeEntry, unsigned> typeMap;

  std::vector<const llvm::Value *> globalValues;
  llvm::DenseMap<const llvm::Value *, unsigned> globalValueMap;

  std::vector<const llvm::Constant *> moduleConstants;
  llvm::DenseMap<const llvm::Constant *, unsigned> moduleConstMap;

  const PointeeTypeMap &PTM;
  // Inferred pointee types for pointer Types (from first usage seen)
  mutable llvm::DenseMap<llvm::Type *, llvm::Type *> inferredPointee;

  /// Per-function-type param type indices. When a FunctionType has pointer
  /// params with different pointee types (e.g., float* and i32*), the generic
  /// typeIdx(ptrType) would return the same index for all. This map stores
  /// the correct per-param type table index for each FunctionType.
  llvm::DenseMap<llvm::FunctionType *, llvm::SmallVector<unsigned, 8>>
      funcTypeParamIndices;

  /// Per-function-type return type index. When a FunctionType returns a
  /// pointer with a non-default pointee (e.g., event_t* for async_copy),
  /// this stores the correct type table index for the return type.
  llvm::DenseMap<llvm::FunctionType *, unsigned> funcTypeReturnIndex;

  ValueEnumerator(llvm::Module &M, const PointeeTypeMap &PTM);

  /// Get type index for a non-pointer type.
  unsigned typeIdx(llvm::Type *T);

  /// Get type index for a pointer type with specific pointee.
  unsigned ptrTypeIdx(llvm::Type *PtrTy, llvm::Type *Pointee);

  /// Get typed pointer index for a Value using PTM inference.
  unsigned ptrTypeIdxForValue(const llvm::Value *V);

  /// Get type index for a global variable's pointer type (uses value type as
  /// pointee).
  unsigned globalPtrTypeIdx(const llvm::GlobalVariable *GV);

  unsigned globalIdx(const llvm::Value *V) const;
  unsigned moduleConstIdx(const llvm::Constant *C) const;
  bool hasModuleConst(const llvm::Constant *C) const;

  /// Default pointee for a pointer type (from inference / fallback).
  llvm::Type *pointeeType(llvm::Type *PtrTy) const;

  /// Pointee for a specific value (PTM → inference → default).
  llvm::Type *pointeeTypeForValue(const llvm::Value *V) const;

  /// Add a constant to the module constant table (and its sub-constants).
  /// Public so that FunctionWriter can ensure aggregate sub-constants exist
  /// in the module constant table before ConstantsWriter references them.
  void addModuleConstant(const llvm::Constant *C);

private:
  unsigned addType(llvm::Type *T);
  unsigned addFunctionType(llvm::FunctionType *FT, const llvm::Function *F);
  unsigned addEntry(TypeEntry E);
  void collectMetadataConstants(const llvm::MDNode *N);
};

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_VALUEENUMERATOR_H
