//===- ValueEnumerator.h - AIR value/type/metadata enumerator ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_VALUEENUMERATOR_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_VALUEENUMERATOR_H

#include "PointeeTypeMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include <vector>

namespace llvm {
namespace metal {

struct TypeEntry {
  llvm::Type *type;
  llvm::Type *pointee;

  bool operator==(const TypeEntry &O) const {
    return type == O.type && pointee == O.pointee;
  }
};

}

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

  std::vector<TypeEntry> types;
  llvm::DenseMap<TypeEntry, unsigned> typeMap;

  std::vector<const llvm::Value *> globalValues;
  llvm::DenseMap<const llvm::Value *, unsigned> globalValueMap;

  std::vector<const llvm::Constant *> moduleConstants;
  llvm::DenseMap<const llvm::Constant *, unsigned> moduleConstMap;

  const PointeeTypeMap &PTM;

  mutable llvm::DenseMap<llvm::Type *, llvm::Type *> inferredPointee;

  llvm::DenseMap<llvm::FunctionType *, llvm::SmallVector<unsigned, 8>>
      funcTypeParamIndices;

  llvm::DenseMap<llvm::FunctionType *, unsigned> funcTypeReturnIndex;

  ValueEnumerator(llvm::Module &M, const PointeeTypeMap &PTM);

  bool frozen = false;

  unsigned typeIdx(llvm::Type *T);

  unsigned ptrTypeIdx(llvm::Type *PtrTy, llvm::Type *Pointee);

  unsigned ptrTypeIdxForValue(const llvm::Value *V);

  unsigned typeIdxForValue(const llvm::Value *V);

  unsigned globalPtrTypeIdx(const llvm::GlobalVariable *GV);

  unsigned globalIdx(const llvm::Value *V) const;
  unsigned moduleConstIdx(const llvm::Constant *C) const;
  bool hasModuleConst(const llvm::Constant *C) const;

  llvm::Type *pointeeType(llvm::Type *PtrTy) const;

  llvm::Type *pointeeTypeForValue(const llvm::Value *V) const;

  void addModuleConstant(const llvm::Constant *C);

private:
  unsigned addType(llvm::Type *T);
  unsigned addFunctionType(llvm::FunctionType *FT, const llvm::Function *F);
  unsigned addEntry(TypeEntry E);
  void collectMetadataConstants(const llvm::MDNode *N);
  void
  collectMetadataConstants(const llvm::MDNode *N,
                           llvm::SmallPtrSetImpl<const llvm::MDNode *> &Seen);
};

}
}

#endif
