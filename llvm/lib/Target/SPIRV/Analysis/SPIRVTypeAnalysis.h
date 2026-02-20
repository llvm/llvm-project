//===- SPIRVTypeAnalysis.h ------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis links a type information to every register/pointer, allowing
// us to legalize type mismatches when required (graphical SPIR-V pointers for
// ex).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVTYPEANALYSIS_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVTYPEANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include <iostream>
#include <optional>
#include <unordered_set>

namespace llvm {
class SPIRVSubtarget;
class MachineFunction;
class MachineModuleInfo;

namespace SPIRV {

// Holds a ConvergenceRegion hierarchy.
class TypeInfo {
  DenseMap<const Value *, Type *> *TypeMap;

public:
  TypeInfo() : TypeMap(nullptr) {}
  TypeInfo(DenseMap<const Value *, Type *> *TypeMap) : TypeMap(TypeMap) {}

  Type *getType(const Value *V) {
    auto It = TypeMap->find(V);
    if (It != TypeMap->end())
      return It->second;

    // In some cases, type deduction is not possible from the IR. This should
    // only happen when handling opaque pointers, otherwise it means the type
    // deduction is broken.
    assert(isOpaqueType(V->getType()));
    return V->getType();
  }

  // Returns true this type contains no opaque pointers (recursively).
  static bool isOpaqueType(const Type *T);
};

} // namespace SPIRV

// Wrapper around the function above to use it with the legacy pass manager.
class SPIRVTypeAnalysisWrapperPass : public ModulePass {
  SPIRV::TypeInfo TI;

public:
  static char ID;

  SPIRVTypeAnalysisWrapperPass();

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  };

  bool runOnModule(Module &F) override;

  SPIRV::TypeInfo &getTypeInfo() { return TI; }
  const SPIRV::TypeInfo &getTypeInfo() const { return TI; }
};

// Wrapper around the function above to use it with the new pass manager.
class SPIRVTypeAnalysis : public AnalysisInfoMixin<SPIRVTypeAnalysis> {
  friend AnalysisInfoMixin<SPIRVTypeAnalysis>;
  static AnalysisKey Key;

public:
  using Result = SPIRV::TypeInfo;

  Result run(Module &F, ModuleAnalysisManager &AM);
};

namespace SPIRV {
TypeInfo getTypeInfo(Module &F);
} // namespace SPIRV

} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVTYPEANALYSIS_H
