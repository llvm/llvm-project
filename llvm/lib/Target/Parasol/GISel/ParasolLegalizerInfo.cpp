//===- ParasolLegalizerInfo.h ----------------------------------------------==//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the Machinelegalizer class for Parasol
//===----------------------------------------------------------------------===//

#include "ParasolLegalizerInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "parasol-legalinfo"

using namespace llvm;

using namespace LegalizeActions;

// Found in the data layout of the target
#define POINTER_SIZE 32

ParasolLegalizerInfo::ParasolLegalizerInfo(const ParasolSubtarget &ST)
    : ST(&ST) {
  using namespace TargetOpcode;

  const LLT p0 = LLT::pointer(0, POINTER_SIZE);
  const LLT s1 = LLT::scalar(1);
  const LLT s8 = LLT::scalar(8);
  const LLT s16 = LLT::scalar(16);
  const LLT s32 = LLT::scalar(32);

  std::initializer_list<LLT> ScalarAndPtrTypesList = {s1, s8, s16, s32, p0};
  SmallVector<LLT, 8> ScalarAndPtrTypesVec(ScalarAndPtrTypesList);

  getActionDefinitionsBuilder({G_STORE, G_LOAD})
      .legalForTypesWithMemDesc({
          {s8, p0, s8, 1},
          {s16, p0, s16, 1},
          {s32, p0, s32, 1},
      });

  getActionDefinitionsBuilder({G_ADD, G_SUB, G_MUL, G_AND, G_OR, G_XOR})
      .legalFor({s1, s8, s16, s32});

  getLegacyLegalizerInfo().computeTables();
  verify(*ST.getInstrInfo());
}
