//===- AMDGPUWaitcntUtils.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUWaitcntUtils.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
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
  case TENSOR_CNT:
    return "TENSOR_CNT";
  case VA_VDST_RD:
    return "VA_VDST_RD";
  case VA_VDST_WR:
    return "VA_VDST_WR";
  case VM_VSRC:
    return "VM_VSRC";
  case NUM_INST_CNTS:
    return "NUM_INST_CNTS";
  }
  llvm_unreachable("Unhandled InstCounterType");
}

HardwareLimits::HardwareLimits(const IsaVersion &IV) {
  bool HasExtendedWaitCounts = IV.Major >= 12;
  if (HasExtendedWaitCounts) {
    LoadcntMax = getLoadcntBitMask(IV);
    DscntMax = getDscntBitMask(IV);
  } else {
    LoadcntMax = getVmcntBitMask(IV);
    DscntMax = getLgkmcntBitMask(IV);
  }
  ExpcntMax = getExpcntBitMask(IV);
  StorecntMax = getStorecntBitMask(IV);
  SamplecntMax = getSamplecntBitMask(IV);
  BvhcntMax = getBvhcntBitMask(IV);
  KmcntMax = getKmcntBitMask(IV);
  XcntMax = getXcntBitMask(IV);
  AsyncMax = getAsynccntBitMask(IV);
  VaVdstMax = DepCtr::getVaVdstBitMask();
  VmVsrcMax = DepCtr::getVmVsrcBitMask();
}

unsigned HardwareLimits::get(InstCounterType T) const {
  switch (T) {
  case AMDGPU::LOAD_CNT:
    return LoadcntMax;
  case AMDGPU::DS_CNT:
    return DscntMax;
  case AMDGPU::EXP_CNT:
    return ExpcntMax;
  case AMDGPU::STORE_CNT:
    return StorecntMax;
  case AMDGPU::SAMPLE_CNT:
    return SamplecntMax;
  case AMDGPU::BVH_CNT:
    return BvhcntMax;
  case AMDGPU::KM_CNT:
    return KmcntMax;
  case AMDGPU::X_CNT:
    return XcntMax;
  case AMDGPU::VA_VDST_RD:
  case AMDGPU::VA_VDST_WR:
    return VaVdstMax;
  case AMDGPU::VM_VSRC:
    return VmVsrcMax;
  default:
    return 0;
  }
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

std::optional<AMDGPU::InstCounterType> counterTypeForInstr(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::S_WAIT_LOADCNT:
    return AMDGPU::LOAD_CNT;
  case AMDGPU::S_WAIT_EXPCNT:
    return AMDGPU::EXP_CNT;
  case AMDGPU::S_WAIT_STORECNT:
    return AMDGPU::STORE_CNT;
  case AMDGPU::S_WAIT_SAMPLECNT:
    return AMDGPU::SAMPLE_CNT;
  case AMDGPU::S_WAIT_BVHCNT:
    return AMDGPU::BVH_CNT;
  case AMDGPU::S_WAIT_DSCNT:
    return AMDGPU::DS_CNT;
  case AMDGPU::S_WAIT_KMCNT:
    return AMDGPU::KM_CNT;
  case AMDGPU::S_WAIT_XCNT:
    return AMDGPU::X_CNT;
  case AMDGPU::S_WAIT_ASYNCCNT:
    return AMDGPU::ASYNC_CNT;
  case AMDGPU::S_WAIT_TENSORCNT:
    return AMDGPU::TENSOR_CNT;
  default:
    return {};
  }
}

} // namespace llvm::AMDGPU
