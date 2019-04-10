//===-- Tapir.h - Tapir Transformations -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
// LoopSpawning - Create a loop spawning pass.
//
Pass *createLoopSpawningPass();

//===----------------------------------------------------------------------===//
//
// LoopSpawningTI - Create a loop spawning pass that uses Task Info.
//
Pass *createLoopSpawningTIPass();

//===----------------------------------------------------------------------===//
//
// SmallBlock - Do SmallBlock Pass
//
FunctionPass *createSmallBlockPass();

//===----------------------------------------------------------------------===//
//
// SyncElimination - TODO
//
FunctionPass *createSyncEliminationPass();

//===----------------------------------------------------------------------===//
//
// RedundantSpawn - Do RedundantSpawn Pass
//
FunctionPass *createRedundantSpawnPass();

//===----------------------------------------------------------------------===//
//
// SpawnRestructure - Do SpawnRestructure Pass
//
FunctionPass *createSpawnRestructurePass();

//===----------------------------------------------------------------------===//
//
// SpawnUnswitch - Do SpawnUnswitch Pass
//
FunctionPass *createSpawnUnswitchPass();

//===----------------------------------------------------------------------===//
//
// LowerTapirToTarget - Lower Tapir constructs to a specified parallel runtime.
//
ModulePass *createLowerTapirToTargetPass();

//===----------------------------------------------------------------------===//
//
//
FunctionPass *createAnalyzeTapirPass();

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
