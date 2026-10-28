//===- AMDGPUWaitcntUtils.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"

namespace llvm::AMDGPU {

static cl::opt<bool> ExpertSchedulingModeFlag(
    "amdgpu-expert-scheduling-mode",
    cl::desc("Enable expert scheduling mode 2 for all functions (GFX12+ only)"),
    cl::init(false), cl::Hidden);

bool isExpertSchedulingMode(const GCNSubtarget &ST, const Function &F) {
  return ST.hasExpertSchedulingMode() &&
         (ExpertSchedulingModeFlag.getNumOccurrences()
              ? ExpertSchedulingModeFlag
              : F.getFnAttribute("amdgpu-expert-scheduling-mode")
                    .getValueAsBool());
}

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

HardwareLimits::HardwareLimits(const CounterLayout &Layout) {
  bool HasExtendedWaitCounts = Layout.LoadcntBits != 0;
  if (HasExtendedWaitCounts) {
    LoadcntMax = getLoadcntBitMask(Layout);
    DscntMax = getDscntBitMask(Layout);
  } else {
    LoadcntMax = getVmcntBitMask(Layout);
    DscntMax = getLgkmcntBitMask(Layout);
  }
  ExpcntMax = getExpcntBitMask(Layout);
  StorecntMax = getStorecntBitMask(Layout);
  SamplecntMax = getSamplecntBitMask(Layout);
  BvhcntMax = getBvhcntBitMask(Layout);
  KmcntMax = getKmcntBitMask(Layout);
  XcntMax = getXcntBitMask(Layout);
  AsyncMax = getAsynccntBitMask(Layout);
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

Waitcnt decodeWaitcnt(const CounterLayout &Layout, unsigned Encoded) {
  Waitcnt Decoded;
  Decoded.set(LOAD_CNT, decodeVmcnt(Layout, Encoded));
  Decoded.set(EXP_CNT, decodeExpcnt(Layout, Encoded));
  Decoded.set(DS_CNT, decodeLgkmcnt(Layout, Encoded));
  return Decoded;
}

unsigned encodeWaitcnt(const CounterLayout &Layout, const Waitcnt &Decoded) {
  return encodeWaitcnt(Layout, Decoded.get(LOAD_CNT), Decoded.get(EXP_CNT),
                       Decoded.get(DS_CNT));
}

Waitcnt decodeLoadcntDscnt(const CounterLayout &Layout, unsigned LoadcntDscnt) {
  Waitcnt Decoded;
  Decoded.set(LOAD_CNT, decodeLoadcnt(Layout, LoadcntDscnt));
  Decoded.set(DS_CNT, decodeDscnt(Layout, LoadcntDscnt));
  return Decoded;
}

Waitcnt decodeStorecntDscnt(const CounterLayout &Layout,
                            unsigned StorecntDscnt) {
  Waitcnt Decoded;
  Decoded.set(STORE_CNT, decodeStorecnt(Layout, StorecntDscnt));
  Decoded.set(DS_CNT, decodeDscnt(Layout, StorecntDscnt));
  return Decoded;
}

unsigned encodeLoadcntDscnt(const CounterLayout &Layout,
                            const Waitcnt &Decoded) {
  return encodeLoadcntDscnt(Layout, Decoded.get(LOAD_CNT), Decoded.get(DS_CNT));
}

unsigned encodeStorecntDscnt(const CounterLayout &Layout,
                             const Waitcnt &Decoded) {
  return encodeStorecntDscnt(Layout, Decoded.get(STORE_CNT),
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
