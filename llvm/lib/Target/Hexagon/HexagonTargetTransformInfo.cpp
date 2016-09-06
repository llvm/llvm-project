//===-- HexagonTargetTransformInfo.cpp - Hexagon specific TTI pass --------===//
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
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "hexagontti"

TargetTransformInfo::PopcntSupportKind
HexagonTTIImpl::getPopcntSupport(unsigned IntTyWidthInBit) const {
  // Return Fast Hardware support as every input  < 64 bits will be promoted
  // to 64 bits.
  return TargetTransformInfo::PSK_FastHardware;
}

// The Hexagon target can unroll loops with run-time trip counts.
void HexagonTTIImpl::getUnrollingPreferences(Loop *L,
                                             TTI::UnrollingPreferences &UP) {
  UP.Runtime = UP.Partial = true;
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
