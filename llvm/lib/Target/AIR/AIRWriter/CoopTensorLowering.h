//===- CoopTensorLowering.h - matmul2d / cooperative_tensor glue -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// All writer plumbing specific to the matmul2d / cooperative_tensor feature
// (env METAL_COOP_MMA, default-off), kept out of the generic always-on
// pipeline: the `__tensorops_impl_` externally_defined section re-tag, the
// tensor-builtin arg→pointee table, and the tensor-handle pinning consulted by
// PointeeTypeMap::inferFromUsage. Each entry no-ops when no tensor-ops symbols
// are present.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_COOPTENSORLOWERING_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_COOPTENSORLOWERING_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Type.h"

namespace llvm {
class Module;
class Value;
class LLVMContext;
namespace metal {
class PointeeTypeMap;

inline constexpr llvm::StringRef kExternallyDefinedSection =
    "air.externally_defined";

void retagTensorOpsExternallyDefined(Module &M);

Type *requiredTensorArgPointee(StringRef Name, unsigned ArgNo,
                               LLVMContext &Ctx);

Type *tensorHandlePointee(StringRef CalleeName, unsigned ArgNo, Value *Ptr,
                          LLVMContext &Ctx);

void fixTensorRuntimeArgTypes(Module &M, PointeeTypeMap &PTM);

}
}

#endif
