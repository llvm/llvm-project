//===- NVVMIntrinsicLowering.h - NVVM op lowering to intrinsics -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares what the NVVM ops use to declare their lowering to LLVM
// intrinsics, such as `getIntrinsicIDAndArgs`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMINTRINSICLOWERING_H_
#define MLIR_DIALECT_LLVMIR_NVVMINTRINSICLOWERING_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Intrinsics.h"

#include <utility>

namespace llvm {
class IRBuilderBase;
class Value;
} // namespace llvm

namespace mlir {
namespace LLVM {
class ModuleTranslation;
} // namespace LLVM

namespace NVVM {
/// A pair type of LLVM's Intrinsic ID and args (which are llvm values).
/// This type is returned by the getIntrinsicIDAndArgs() methods.
using IDArgPair =
    std::pair<llvm::Intrinsic::ID, llvm::SmallVector<llvm::Value *>>;
} // namespace NVVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_NVVMINTRINSICLOWERING_H_
