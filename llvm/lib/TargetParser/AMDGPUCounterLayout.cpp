//===-- AMDGPUCounterLayout.cpp - AMDGPU counter field layout -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/AMDGPUCounterLayout.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::AMDGPU;

// Field positions no GPU has changed.
static constexpr unsigned VmcntHiShift = 14;
static constexpr unsigned ExpcntBits = 3;
static constexpr unsigned DscntShift = 0;
static constexpr unsigned HoldCntShift = 7;

static unsigned getBitMask(unsigned Shift, unsigned Width) {
  return maskTrailingOnes<unsigned>(Width) << Shift;
}

static unsigned packBits(unsigned Src, unsigned Dst, unsigned Shift,
                         unsigned Width) {
  unsigned Mask = getBitMask(Shift, Width);
  return ((Src << Shift) & Mask) | (Dst & ~Mask);
}

static unsigned unpackBits(unsigned Src, unsigned Shift, unsigned Width) {
  return (Src & getBitMask(Shift, Width)) >> Shift;
}

unsigned AMDGPU::getVmcntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.VmcntLoBits + L.VmcntHiBits);
}

unsigned AMDGPU::getExpcntBitMask(const CounterLayout &) {
  return maskTrailingOnes<unsigned>(ExpcntBits);
}

unsigned AMDGPU::getLgkmcntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.LgkmcntBits);
}

unsigned AMDGPU::getWaitcntBitMask(const CounterLayout &L) {
  return getBitMask(L.VmcntLoShift, L.VmcntLoBits) |
         getBitMask(L.ExpcntShift, ExpcntBits) |
         getBitMask(L.LgkmcntShift, L.LgkmcntBits) |
         getBitMask(VmcntHiShift, L.VmcntHiBits);
}

unsigned AMDGPU::getLoadcntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.LoadcntBits);
}

unsigned AMDGPU::getSamplecntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.SamplecntBits);
}

unsigned AMDGPU::getBvhcntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.BvhcntBits);
}

unsigned AMDGPU::getAsynccntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.AsynccntBits);
}

unsigned AMDGPU::getDscntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.DscntBits);
}

unsigned AMDGPU::getKmcntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.KmcntBits);
}

unsigned AMDGPU::getXcntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.XcntBits);
}

unsigned AMDGPU::getStorecntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.StorecntBits);
}

unsigned AMDGPU::getBarrierMemberCountBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.BarrierMemberCountBits);
}

unsigned AMDGPU::decodeVmcnt(const CounterLayout &L, unsigned Waitcnt) {
  unsigned Lo = unpackBits(Waitcnt, L.VmcntLoShift, L.VmcntLoBits);
  unsigned Hi = unpackBits(Waitcnt, VmcntHiShift, L.VmcntHiBits);
  return Lo | Hi << L.VmcntLoBits;
}

unsigned AMDGPU::decodeExpcnt(const CounterLayout &L, unsigned Waitcnt) {
  return unpackBits(Waitcnt, L.ExpcntShift, ExpcntBits);
}

unsigned AMDGPU::decodeLgkmcnt(const CounterLayout &L, unsigned Waitcnt) {
  return unpackBits(Waitcnt, L.LgkmcntShift, L.LgkmcntBits);
}

unsigned AMDGPU::decodeLoadcnt(const CounterLayout &L, unsigned Waitcnt) {
  return unpackBits(Waitcnt, L.LoadStorecntShift, L.LoadcntBits);
}

unsigned AMDGPU::decodeStorecnt(const CounterLayout &L, unsigned Waitcnt) {
  return unpackBits(Waitcnt, L.LoadStorecntShift, L.StorecntBits);
}

unsigned AMDGPU::decodeDscnt(const CounterLayout &L, unsigned Waitcnt) {
  return unpackBits(Waitcnt, DscntShift, L.DscntBits);
}

void AMDGPU::decodeWaitcnt(const CounterLayout &L, unsigned Waitcnt,
                           unsigned &Vmcnt, unsigned &Expcnt,
                           unsigned &Lgkmcnt) {
  Vmcnt = decodeVmcnt(L, Waitcnt);
  Expcnt = decodeExpcnt(L, Waitcnt);
  Lgkmcnt = decodeLgkmcnt(L, Waitcnt);
}

unsigned AMDGPU::encodeVmcnt(const CounterLayout &L, unsigned Waitcnt,
                             unsigned Vmcnt) {
  Waitcnt = packBits(Vmcnt, Waitcnt, L.VmcntLoShift, L.VmcntLoBits);
  return packBits(Vmcnt >> L.VmcntLoBits, Waitcnt, VmcntHiShift, L.VmcntHiBits);
}

unsigned AMDGPU::encodeExpcnt(const CounterLayout &L, unsigned Waitcnt,
                              unsigned Expcnt) {
  return packBits(Expcnt, Waitcnt, L.ExpcntShift, ExpcntBits);
}

unsigned AMDGPU::encodeLgkmcnt(const CounterLayout &L, unsigned Waitcnt,
                               unsigned Lgkmcnt) {
  return packBits(Lgkmcnt, Waitcnt, L.LgkmcntShift, L.LgkmcntBits);
}

unsigned AMDGPU::encodeWaitcnt(const CounterLayout &L, unsigned Vmcnt,
                               unsigned Expcnt, unsigned Lgkmcnt) {
  unsigned Waitcnt = getWaitcntBitMask(L);
  Waitcnt = encodeVmcnt(L, Waitcnt, Vmcnt);
  Waitcnt = encodeExpcnt(L, Waitcnt, Expcnt);
  return encodeLgkmcnt(L, Waitcnt, Lgkmcnt);
}

// Combined S_WAIT_*CNT_DSCNT immediate: \p Cnt at LoadStorecntShift with
// \p CntBits, Dscnt at DscntShift.
static unsigned encodeCntDscnt(const CounterLayout &L, unsigned Cnt,
                               unsigned CntBits, unsigned Dscnt) {
  unsigned Waitcnt = getBitMask(L.LoadStorecntShift, CntBits) |
                     getBitMask(DscntShift, L.DscntBits);
  Waitcnt = packBits(Cnt, Waitcnt, L.LoadStorecntShift, CntBits);
  return packBits(Dscnt, Waitcnt, DscntShift, L.DscntBits);
}

unsigned AMDGPU::encodeLoadcntDscnt(const CounterLayout &L, unsigned Loadcnt,
                                    unsigned Dscnt) {
  return encodeCntDscnt(L, Loadcnt, L.LoadcntBits, Dscnt);
}

unsigned AMDGPU::encodeStorecntDscnt(const CounterLayout &L, unsigned Storecnt,
                                     unsigned Dscnt) {
  return encodeCntDscnt(L, Storecnt, L.StorecntBits, Dscnt);
}

unsigned AMDGPU::DepCtr::getHoldCntBitMask(const CounterLayout &L) {
  return maskTrailingOnes<unsigned>(L.HoldCntBits);
}

unsigned AMDGPU::DepCtr::decodeFieldHoldCnt(unsigned Encoded,
                                            const CounterLayout &L) {
  return unpackBits(Encoded, HoldCntShift, L.HoldCntBits);
}

unsigned AMDGPU::DepCtr::encodeFieldHoldCnt(unsigned Encoded, unsigned HoldCnt,
                                            const CounterLayout &L) {
  return packBits(HoldCnt, Encoded, HoldCntShift, L.HoldCntBits);
}
