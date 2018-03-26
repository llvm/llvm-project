//===- HexagonTargetTransformInfo.cpp - Hexagon specific TTI pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// Hexagon target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "HexagonTargetTransformInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/User.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "hexagontti"

static cl::opt<bool> EmitLookupTables("hexagon-emit-lookup-tables",
  cl::init(true), cl::Hidden,
  cl::desc("Control lookup table emission on Hexagon target"));

TargetTransformInfo::PopcntSupportKind
HexagonTTIImpl::getPopcntSupport(unsigned IntTyWidthInBit) const {
  // Return Fast Hardware support as every input  < 64 bits will be promoted
  // to 64 bits.
  return TargetTransformInfo::PSK_FastHardware;
}

// The Hexagon target can unroll loops with run-time trip counts.
void HexagonTTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                             TTI::UnrollingPreferences &UP) {
  UP.Runtime = UP.Partial = true;
}

bool HexagonTTIImpl::shouldFavorPostInc() const {
  return true;
}

unsigned HexagonTTIImpl::getNumberOfRegisters(bool vector) const {
  return vector ? 0 : 32;
}

unsigned HexagonTTIImpl::getPrefetchDistance() const {
  return getST()->getL1PrefetchDistance();
}

unsigned HexagonTTIImpl::getCacheLineSize() const {
  return getST()->getL1CacheLineSize();
}

int HexagonTTIImpl::getUserCost(const User *U,
                                ArrayRef<const Value *> Operands) {
  auto isCastFoldedIntoLoad = [](const CastInst *CI) -> bool {
    if (!CI->isIntegerCast())
      return false;
    const LoadInst *LI = dyn_cast<const LoadInst>(CI->getOperand(0));
    // Technically, this code could allow multiple uses of the load, and
    // check if all the uses are the same extension operation, but this
    // should be sufficient for most cases.
    if (!LI || !LI->hasOneUse())
      return false;

    // Only extensions from an integer type shorter than 32-bit to i32
    // can be folded into the load.
    unsigned SBW = CI->getSrcTy()->getIntegerBitWidth();
    unsigned DBW = CI->getDestTy()->getIntegerBitWidth();
    return DBW == 32 && (SBW < DBW);
  };

  if (const CastInst *CI = dyn_cast<const CastInst>(U))
    if (isCastFoldedIntoLoad(CI))
      return TargetTransformInfo::TCC_Free;
  return BaseT::getUserCost(U, Operands);
}

bool HexagonTTIImpl::shouldBuildLookupTables() const {
  return EmitLookupTables;
}
