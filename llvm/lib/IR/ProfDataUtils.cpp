//===- ProfDataUtils.cpp - Utility functions for MD_prof Metadata ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for working with Profiling Metadata.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ProfDataUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace {

// MD_prof nodes have the following layout
//
// In general:
// { String name,         Array of i32   }
//
// In terms of Types:
// { MDString,            [i32, i32, ...]}
//
// Concretely for Branch Weights
// { "branch_weights",    [i32 1, i32 10000]}
//
// We maintain some constants here to ensure that we access the branch weights
// correctly, and can change the behavior in the future if the layout changes

// The index at which the weights vector starts
constexpr unsigned WeightsIdx = 1;

// the minimum number of operands for MD_prof nodes with branch weights
constexpr unsigned MinBWOps = 3;

// We may want to add support for other MD_prof types, so provide an abstraction
// for checking the metadata type.
bool isTargetMD(const MDNode *ProfData, const char *Name, unsigned MinOps) {
  // TODO: This routine may be simplified if MD_prof used an enum instead of a
  // string to differentiate the types of MD_prof nodes.
  if (!ProfData || !Name || MinOps < 2)
    return false;

  unsigned NOps = ProfData->getNumOperands();
  if (NOps < MinOps)
    return false;

  auto *ProfDataName = dyn_cast<MDString>(ProfData->getOperand(0));
  if (!ProfDataName)
    return false;

  return ProfDataName->getString().equals(Name);
}

} // namespace

namespace llvm {

bool hasProfMD(const Instruction &I) {
  return nullptr != I.getMetadata(LLVMContext::MD_prof);
}

bool isBranchWeightMD(const MDNode *ProfileData) {
  return isTargetMD(ProfileData, "branch_weights", MinBWOps);
}

bool hasBranchWeightMD(const Instruction &I) {
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  return isBranchWeightMD(ProfileData);
}

bool hasValidBranchWeightMD(const Instruction &I) {
  return getValidBranchWeightMDNode(I);
}

MDNode *getBranchWeightMDNode(const Instruction &I) {
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  if (!isBranchWeightMD(ProfileData))
    return nullptr;
  return ProfileData;
}

MDNode *getValidBranchWeightMDNode(const Instruction &I) {
  auto *ProfileData = getBranchWeightMDNode(I);
  if (ProfileData && ProfileData->getNumOperands() == 1 + I.getNumSuccessors())
    return ProfileData;
  return nullptr;
}

void extractFromBranchWeightMD(const MDNode *ProfileData,
                               SmallVectorImpl<uint32_t> &Weights) {
  assert(isBranchWeightMD(ProfileData) && "wrong metadata");

  unsigned NOps = ProfileData->getNumOperands();
  assert(WeightsIdx < NOps && "Weights Index must be less than NOps.");
  Weights.resize(NOps - WeightsIdx);

  for (unsigned Idx = WeightsIdx, E = NOps; Idx != E; ++Idx) {
    ConstantInt *Weight =
        mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(Idx));
    assert(Weight && "Malformed branch_weight in MD_prof node");
    assert(Weight->getValue().getActiveBits() <= 32 &&
           "Too many bits for uint32_t");
    Weights[Idx - WeightsIdx] = Weight->getZExtValue();
  }
}

bool extractBranchWeights(const MDNode *ProfileData,
                          SmallVectorImpl<uint32_t> &Weights) {
  if (!isBranchWeightMD(ProfileData))
    return false;
  extractFromBranchWeightMD(ProfileData, Weights);
  return true;
}

bool extractBranchWeights(const Instruction &I,
                          SmallVectorImpl<uint32_t> &Weights) {
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  return extractBranchWeights(ProfileData, Weights);
}

bool extractBranchWeights(const Instruction &I, uint64_t &TrueVal,
                          uint64_t &FalseVal) {
  assert((I.getOpcode() == Instruction::Br ||
          I.getOpcode() == Instruction::Select) &&
         "Looking for branch weights on something besides branch, select, or "
         "switch");

  SmallVector<uint32_t, 2> Weights;
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  if (!extractBranchWeights(ProfileData, Weights))
    return false;

  if (Weights.size() > 2)
    return false;

  TrueVal = Weights[0];
  FalseVal = Weights[1];
  return true;
}

bool extractProfTotalWeight(const MDNode *ProfileData, uint64_t &TotalVal) {
  TotalVal = 0;
  if (!ProfileData)
    return false;

  auto *ProfDataName = dyn_cast<MDString>(ProfileData->getOperand(0));
  if (!ProfDataName)
    return false;

  if (ProfDataName->getString().equals("branch_weights")) {
    for (unsigned Idx = 1; Idx < ProfileData->getNumOperands(); Idx++) {
      auto *V = mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(Idx));
      assert(V && "Malformed branch_weight in MD_prof node");
      TotalVal += V->getValue().getZExtValue();
    }
    return true;
  }

  if (ProfDataName->getString().equals("VP") &&
      ProfileData->getNumOperands() > 3) {
    TotalVal = mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(2))
                   ->getValue()
                   .getZExtValue();
    return true;
  }
  return false;
}

bool extractProfTotalWeight(const Instruction &I, uint64_t &TotalVal) {
  return extractProfTotalWeight(I.getMetadata(LLVMContext::MD_prof), TotalVal);
}

} // namespace llvm
