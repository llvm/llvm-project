//===- llvm/IR/ProfDataUtils.h - Profiling Metadata Utilities ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the declarations for profiling metadata utility
/// functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PROFDATAUTILS_H
#define LLVM_IR_PROFDATAUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
struct MDProfLabels {
  LLVM_ABI static const char *BranchWeights;
  LLVM_ABI static const char *ValueProfile;
  LLVM_ABI static const char *FunctionEntryCount;
  LLVM_ABI static const char *SyntheticFunctionEntryCount;
  LLVM_ABI static const char *ExpectedBranchWeights;
  LLVM_ABI static const char *UnknownBranchWeightsMarker;
};

/// Profile-based loop metadata that should be accessed only by using
/// \c llvm::getLoopEstimatedTripCount and \c llvm::setLoopEstimatedTripCount.
LLVM_ABI extern const char *LLVMLoopEstimatedTripCount;

/// Checks if an Instruction has MD_prof Metadata
LLVM_ABI bool hasProfMD(const Instruction &I);

/// Checks if an MDNode contains Branch Weight Metadata
LLVM_ABI bool isBranchWeightMD(const MDNode *ProfileData);

/// Checks if an MDNode contains value profiling Metadata
LLVM_ABI bool isValueProfileMD(const MDNode *ProfileData);

/// Checks if an instructions has Branch Weight Metadata
///
/// \param I The instruction to check
/// \returns True if I has an MD_prof node containing Branch Weights. False
/// otherwise.
LLVM_ABI bool hasBranchWeightMD(const Instruction &I);

/// Checks if an instructions has valid Branch Weight Metadata
///
/// \param I The instruction to check
/// \returns True if I has an MD_prof node containing valid Branch Weights,
/// i.e., one weight for each successor. False otherwise.
LLVM_ABI bool hasValidBranchWeightMD(const Instruction &I);

/// Get the branch weights metadata node
///
/// \param I The Instruction to get the weights from.
/// \returns A pointer to I's branch weights metadata node, if it exists.
/// Nullptr otherwise.
LLVM_ABI MDNode *getBranchWeightMDNode(const Instruction &I);

/// Get the valid branch weights metadata node
///
/// \param I The Instruction to get the weights from.
/// \returns A pointer to I's valid branch weights metadata node, if it exists.
/// Nullptr otherwise.
LLVM_ABI MDNode *getValidBranchWeightMDNode(const Instruction &I);

/// Check if Branch Weight Metadata has an "expected" field from an llvm.expect*
/// intrinsic
LLVM_ABI bool hasBranchWeightOrigin(const Instruction &I);

/// Check if Branch Weight Metadata has an "expected" field from an llvm.expect*
/// intrinsic
LLVM_ABI bool hasBranchWeightOrigin(const MDNode *ProfileData);

/// Return the offset to the first branch weight data
LLVM_ABI unsigned getBranchWeightOffset(const MDNode *ProfileData);

LLVM_ABI unsigned getNumBranchWeights(const MDNode &ProfileData);

/// Extract branch weights from MD_prof metadata
///
/// \param ProfileData A pointer to an MDNode.
/// \param [out] Weights An output vector to fill with branch weights
/// \returns True if weights were extracted, False otherwise. When false Weights
/// will be cleared.
LLVM_ABI bool extractBranchWeights(const MDNode *ProfileData,
                                   SmallVectorImpl<uint32_t> &Weights);

/// Faster version of extractBranchWeights() that skips checks and must only
/// be called with "branch_weights" metadata nodes. Supports uint32_t.
LLVM_ABI void extractFromBranchWeightMD32(const MDNode *ProfileData,
                                          SmallVectorImpl<uint32_t> &Weights);

/// Faster version of extractBranchWeights() that skips checks and must only
/// be called with "branch_weights" metadata nodes. Supports uint64_t.
LLVM_ABI void extractFromBranchWeightMD64(const MDNode *ProfileData,
                                          SmallVectorImpl<uint64_t> &Weights);

/// Extract branch weights attatched to an Instruction
///
/// \param I The Instruction to extract weights from.
/// \param [out] Weights An output vector to fill with branch weights
/// \returns True if weights were extracted, False otherwise. When false Weights
/// will be cleared.
LLVM_ABI bool extractBranchWeights(const Instruction &I,
                                   SmallVectorImpl<uint32_t> &Weights);

/// Extract branch weights from a conditional branch or select Instruction.
///
/// \param I The instruction to extract branch weights from.
/// \param [out] TrueVal will contain the branch weight for the True branch
/// \param [out] FalseVal will contain the branch weight for the False branch
/// \returns True on success with profile weights filled in. False if no
/// metadata or invalid metadata was found.
LLVM_ABI bool extractBranchWeights(const Instruction &I, uint64_t &TrueVal,
                                   uint64_t &FalseVal);

/// Retrieve the total of all weights from MD_prof data.
///
/// \param ProfileData The profile data to extract the total weight from
/// \param [out] TotalWeights input variable to fill with total weights
/// \returns True on success with profile total weights filled in. False if no
/// metadata was found.
LLVM_ABI bool extractProfTotalWeight(const MDNode *ProfileData,
                                     uint64_t &TotalWeights);

/// Retrieve the total of all weights from an instruction.
///
/// \param I The instruction to extract the total weight from
/// \param [out] TotalWeights input variable to fill with total weights
/// \returns True on success with profile total weights filled in. False if no
/// metadata was found.
LLVM_ABI bool extractProfTotalWeight(const Instruction &I,
                                     uint64_t &TotalWeights);

/// Create a new `branch_weights` metadata node and add or overwrite
/// a `prof` metadata reference to instruction `I`.
/// \param I the Instruction to set branch weights on.
/// \param Weights an array of weights to set on instruction I.
/// \param IsExpected were these weights added from an llvm.expect* intrinsic.
LLVM_ABI void setBranchWeights(Instruction &I, ArrayRef<uint32_t> Weights,
                               bool IsExpected);

/// downscale the given weights preserving the ratio. If the maximum value is
/// not already known and not provided via \param KnownMaxCount , it will be
/// obtained from \param Weights.
LLVM_ABI SmallVector<uint32_t>
downscaleWeights(ArrayRef<uint64_t> Weights,
                 std::optional<uint64_t> KnownMaxCount = std::nullopt);

/// Calculate what to divide by to scale counts.
///
/// Given the maximum count, calculate a divisor that will scale all the
/// weights to strictly less than std::numeric_limits<uint32_t>::max().
inline uint64_t calculateCountScale(uint64_t MaxCount) {
  return MaxCount < std::numeric_limits<uint32_t>::max()
             ? 1
             : MaxCount / std::numeric_limits<uint32_t>::max() + 1;
}

/// Scale an individual branch count.
///
/// Scale a 64-bit weight down to 32-bits using \c Scale.
///
inline uint32_t scaleBranchCount(uint64_t Count, uint64_t Scale) {
  uint64_t Scaled = Count / Scale;
  assert(Scaled <= std::numeric_limits<uint32_t>::max() && "overflow 32-bits");
  return Scaled;
}

/// Specify that the branch weights for this terminator cannot be known at
/// compile time. This should only be called by passes, and never as a default
/// behavior in e.g. MDBuilder. The goal is to use this info to validate passes
/// do not accidentally drop profile info, and this API is called in cases where
/// the pass explicitly cannot provide that info. Defaulting it in would hide
/// bugs where the pass forgets to transfer over or otherwise specify profile
/// info. Use `PassName` to capture the pass name (i.e. DEBUG_TYPE) for
/// debuggability.
LLVM_ABI void setExplicitlyUnknownBranchWeights(Instruction &I,
                                                StringRef PassName);

/// Analogous to setExplicitlyUnknownBranchWeights, but for functions and their
/// entry counts.
LLVM_ABI void setExplicitlyUnknownFunctionEntryCount(Function &F,
                                                     StringRef PassName);

LLVM_ABI bool isExplicitlyUnknownProfileMetadata(const MDNode &MD);
LLVM_ABI bool hasExplicitlyUnknownBranchWeights(const Instruction &I);

/// Scaling the profile data attached to 'I' using the ratio of S/T.
LLVM_ABI void scaleProfData(Instruction &I, uint64_t S, uint64_t T);

} // namespace llvm
#endif
