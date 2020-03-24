//===- Tapir.h - Tapir Transformations --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Tapir transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_H
#define LLVM_TRANSFORMS_TAPIR_H

namespace llvm {
class Pass;
class ModulePass;
class FunctionPass;
enum class TapirTargetID;

//===----------------------------------------------------------------------===//
//
// LoopSpawningTI - Create a loop spawning pass that uses Task Info.
//
Pass *createLoopSpawningTIPass();

//===----------------------------------------------------------------------===//
//
// LowerTapirToTarget - Lower Tapir constructs to a specified parallel runtime.
//
ModulePass *createLowerTapirToTargetPass();

//===----------------------------------------------------------------------===//
//
// TaskSimplify - Simplify Tapir tasks
//
FunctionPass *createTaskSimplifyPass();

//===----------------------------------------------------------------------===//
//
// DRFScopedNoAlias - Add scoped-noalias information based on DRF assumption
//
FunctionPass *createDRFScopedNoAliasWrapperPass();

//===----------------------------------------------------------------------===//
//
// LoopStripMinePass - Stripmine Tapir loops
//
Pass *createLoopStripMinePass(int Count = -1);

//===----------------------------------------------------------------------===//
//
// SerializeSmallTasksPass - Serialize small Tapir tasks
//
FunctionPass *createSerializeSmallTasksPass();

} // End llvm namespace

#endif
