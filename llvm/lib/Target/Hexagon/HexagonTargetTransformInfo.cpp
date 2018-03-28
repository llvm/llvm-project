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

static cl::opt<bool> HexagonAutoHVX("hexagon-autohvx", cl::init(false),
  cl::Hidden, cl::desc("Enable loop vectorizer for HVX"));

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

unsigned HexagonTTIImpl::getNumberOfRegisters(bool Vector) const {
  if (Vector)
    return HexagonAutoHVX && getST()->useHVXOps() ? 32 : 0;
  return 32;
}

unsigned HexagonTTIImpl::getMaxInterleaveFactor(unsigned VF) {
  return HexagonAutoHVX && getST()->useHVXOps() ? 64 : 0;
}

unsigned HexagonTTIImpl::getRegisterBitWidth(bool Vector) const {
  return Vector ? getMinVectorRegisterBitWidth() : 32;
}

unsigned HexagonTTIImpl::getMinVectorRegisterBitWidth() const {
  return getST()->useHVXOps() ? getST()->getVectorLength()*8 : 0;
}

unsigned HexagonTTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
      unsigned Alignment, unsigned AddressSpace, const Instruction *I) {
  if (Opcode == Instruction::Load && Src->isVectorTy()) {
    VectorType *VecTy = cast<VectorType>(Src);
    unsigned VecWidth = VecTy->getBitWidth();
    if (VecWidth > 64) {
      // Assume that vectors longer than 64 bits are meant for HVX.
      if (getNumberOfRegisters(true) > 0) {
        if (VecWidth % getRegisterBitWidth(true) == 0)
          return 1;
      }
      unsigned AlignWidth = 8 * std::max(1u, Alignment);
      unsigned NumLoads = alignTo(VecWidth, AlignWidth) / AlignWidth;
      return 3*NumLoads;
    }
  }
  return BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace, I);
}

unsigned HexagonTTIImpl::getPrefetchDistance() const {
  return getST()->getL1PrefetchDistance();
}

unsigned HexagonTTIImpl::getCacheLineSize() const {
  return getST()->getL1CacheLineSize();
}

int HexagonTTIImpl::getUserCost(const User *U,
                                ArrayRef<const Value *> Operands) {
  auto isCastFoldedIntoLoad = [this](const CastInst *CI) -> bool {
    if (!CI->isIntegerCast())
      return false;
    // Only extensions from an integer type shorter than 32-bit to i32
    // can be folded into the load.
    const DataLayout &DL = getDataLayout();
    unsigned SBW = DL.getTypeSizeInBits(CI->getSrcTy());
    unsigned DBW = DL.getTypeSizeInBits(CI->getDestTy());
    if (DBW != 32 || SBW >= DBW)
      return false;

    const LoadInst *LI = dyn_cast<const LoadInst>(CI->getOperand(0));
    // Technically, this code could allow multiple uses of the load, and
    // check if all the uses are the same extension operation, but this
    // should be sufficient for most cases.
    return LI && LI->hasOneUse();
  };

  if (const CastInst *CI = dyn_cast<const CastInst>(U))
    if (isCastFoldedIntoLoad(CI))
      return TargetTransformInfo::TCC_Free;
  return BaseT::getUserCost(U, Operands);
}

bool HexagonTTIImpl::shouldBuildLookupTables() const {
  return EmitLookupTables;
}
