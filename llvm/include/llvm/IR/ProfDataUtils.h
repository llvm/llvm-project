#ifndef LLVM_IR_PROFDATAUTILS_H
#define LLVM_IR_PROFDATAUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Metadata.h"

namespace llvm {

/// Checks if an Instruction has MD_prof Metadata
bool hasProfMD(const Instruction &I);

/// Checks if an MDNode contains Branch Weight Metadata
bool isBranchWeightMD(const MDNode *ProfileData);

/// Checks if an instructions has Branch Weight Metadata
///
/// \param I The instruction to check
/// \return True if I has an MD_prof node containing Branch Weights. False
/// otherwise.
bool hasBranchWeightMD(const Instruction &I);

/// Extract branch weights from MD_prof metadata
///
/// \param ProfileData A pointer to an MDNode.
/// \param Weights An output vector to fill with branch weights
/// \return True if weights were extracted, False otherwise. When false Weights
/// will be cleared.
bool extractBranchWeights(const MDNode *ProfileData,
                          SmallVectorImpl<uint32_t> &Weights);

/// Extract branch weights attatched to an Instruction
///
/// \param I The Instruction to extract weights from.
/// \param Weights An output vector to fill with branch weights
/// \return True if weights were extracted, False otherwise. When false Weights
/// will be cleared.
bool extractBranchWeights(const Instruction &I,
                          SmallVectorImpl<uint32_t> &Weights);

/// Retrieve the raw weight values of a conditional branch or select.
/// Returns true on success with profile weights filled in.
/// Returns false if no metadata or invalid metadata was found.
bool extractBranchWeights(const Instruction &I, uint64_t &TrueVal,
                          uint64_t &FalseVal);

/// Retrieve the total of all weights from MD_prof data.
///
/// \param ProfileData The profile data to extract the total weight from
/// \param TotalWeights input variable to fill with total weights
/// \return true on success with profile total weights filled in.
/// \return false if no metadata was found.
bool extractProfTotalWeight(const MDNode *ProfileData, uint64_t &TotalWeights);

} // namespace llvm
#endif
