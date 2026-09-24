//===- MLRegAllocEvictAdvisor.cpp - ML eviction advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Function declarations of utilities related to feature extraction for unit
// testing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLREGALLOCEVICTIONADVISOR_H
#define LLVM_CODEGEN_MLREGALLOCEVICTIONADVISOR_H

#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Support/Compiler.h"
#include <map>

namespace llvm {

// LRStartEndInfo contains the start and end of a specific live range as
// slot indices as well as storing the index of the physical register it
// is assigned to (or 1 above the phys reg count if its the candidate).
// Used when extracting per-instruction features in the context of a
// specific eviction problem.
struct LRStartEndInfo {
  SlotIndex Begin;
  SlotIndex End;
  size_t Pos = 0;
};

// Logically, we can think of the feature set given to the evaluator as a 2D
// matrix. The rows are the features (see next). The columns correspond to the
// interferences. We treat the candidate virt reg as an 'interference', too, as
// its feature set is the same as that of the interferring ranges. So we'll have
// one column per allocation order slot, plus one, and by convention, we will
// use the last column for the virt reg seeking allocation.
// The AOT and reference models bake this width in and do not expose it, so
// changing it requires regenerating them. The interactive channel derives its
// own width from the target instead.
static const int64_t CompiledModelNumColumns = 33;

// The number of instructions that a specific live range might have is variable,
// but we're passing in a single matrix of instructions and tensorflow saved
// models only support a fixed input size, so we have to cap the number of
// instructions that can be passed along. The specific value was derived from
// experimentation such that the majority of eviction problems would be
// completely covered.
static const int ModelMaxSupportedInstructionCount = 300;

// When extracting per-instruction features, the advisor will currently create
// a vector of size ModelMaxSupportedInstructionCount to hold the opcodes of the
// instructions relevant to the eviction problem, and a NumberOfInterferences *
// ModelMaxSupportedInstructionCount matrix that maps LRs to the instructions
// that they span.
static const std::vector<int64_t> InstructionsShape{
    1, ModelMaxSupportedInstructionCount};
static const std::vector<int64_t> InstructionsMappingShape{
    1, CompiledModelNumColumns, ModelMaxSupportedInstructionCount};

// When extracting mappings between MBBs and individual instructions, we create
// a vector of MBB frequencies, currently of size 100, which was a value
// determined through experimentation to encompass the vast majority of eviction
// problems. The actual mapping is the same shape as the instruction opcodes
// vector.
static const int64_t ModelMaxSupportedMBBCount = 100;
static const std::vector<int64_t> MBBFrequencyShape{1,
                                                    ModelMaxSupportedMBBCount};

} // namespace llvm

#endif // LLVM_CODEGEN_MLREGALLOCEVICTIONADVISOR_H
