//===- AMDGPUWaitcntUtils.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUWaitcntUtils.h"
#include "Utils/AMDGPUBaseInfo.h"

namespace llvm::AMDGPU {

iota_range<InstCounterType> inst_counter_types(InstCounterType MaxCounter) {
  return enum_seq(LOAD_CNT, MaxCounter);
}

StringLiteral getInstCounterName(InstCounterType T) {
  switch (T) {
  case LOAD_CNT:
    return "LOAD_CNT";
  case DS_CNT:
    return "DS_CNT";
  case EXP_CNT:
    return "EXP_CNT";
  case STORE_CNT:
    return "STORE_CNT";
  case SAMPLE_CNT:
    return "SAMPLE_CNT";
  case BVH_CNT:
    return "BVH_CNT";
  case KM_CNT:
    return "KM_CNT";
  case X_CNT:
    return "X_CNT";
  case ASYNC_CNT:
    return "ASYNC_CNT";
  case VA_VDST:
    return "VA_VDST";
  case VM_VSRC:
    return "VM_VSRC";
  case NUM_INST_CNTS:
    return "NUM_INST_CNTS";
  }
  llvm_unreachable("Unhandled InstCounterType");
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void Waitcnt::dump() const { dbgs() << *this << '\n'; }
#endif

Waitcnt decodeWaitcnt(const IsaVersion &Version, unsigned Encoded) {
  Waitcnt Decoded;
  Decoded.set(LOAD_CNT, decodeVmcnt(Version, Encoded));
  Decoded.set(EXP_CNT, decodeExpcnt(Version, Encoded));
  Decoded.set(DS_CNT, decodeLgkmcnt(Version, Encoded));
  return Decoded;
}

unsigned encodeWaitcnt(const IsaVersion &Version, const Waitcnt &Decoded) {
  return encodeWaitcnt(Version, Decoded.get(LOAD_CNT), Decoded.get(EXP_CNT),
                       Decoded.get(DS_CNT));
}

Waitcnt decodeLoadcntDscnt(const IsaVersion &Version, unsigned LoadcntDscnt) {
  Waitcnt Decoded;
  Decoded.set(LOAD_CNT, decodeLoadcnt(Version, LoadcntDscnt));
  Decoded.set(DS_CNT, decodeDscnt(Version, LoadcntDscnt));
  return Decoded;
}

Waitcnt decodeStorecntDscnt(const IsaVersion &Version, unsigned StorecntDscnt) {
  Waitcnt Decoded;
  Decoded.set(STORE_CNT, decodeStorecnt(Version, StorecntDscnt));
  Decoded.set(DS_CNT, decodeDscnt(Version, StorecntDscnt));
  return Decoded;
}

unsigned encodeLoadcntDscnt(const IsaVersion &Version, const Waitcnt &Decoded) {
  return encodeLoadcntDscnt(Version, Decoded.get(LOAD_CNT),
                            Decoded.get(DS_CNT));
}

unsigned encodeStorecntDscnt(const IsaVersion &Version,
                             const Waitcnt &Decoded) {
  return encodeStorecntDscnt(Version, Decoded.get(STORE_CNT),
                             Decoded.get(DS_CNT));
}

} // namespace llvm::AMDGPU
