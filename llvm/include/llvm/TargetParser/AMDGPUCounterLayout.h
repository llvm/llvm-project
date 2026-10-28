//===-- AMDGPUCounterLayout.h - AMDGPU counter field layout -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bit layout of the AMDGPU wait counters in S_WAITCNT-style immediates, the
// DepCtr HoldCnt field and the barrier member count. Field values come from
// numeric subtarget features; see CounterFieldFeature in GCNFeatures.td.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_AMDGPUCOUNTERLAYOUT_H
#define LLVM_TARGETPARSER_AMDGPUCOUNTERLAYOUT_H

#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <tuple>

namespace llvm::AMDGPU {

/// Bit shift / width of each counter field that varies across GPUs. A zero
/// width means the counter does not exist. Defaults are the SI layout, used by
/// pseudo and unknown targets; the TableGen emitter reads them from here.
struct CounterLayout {
  uint8_t VmcntLoShift = 0;
  uint8_t VmcntLoBits = 4;
  uint8_t VmcntHiBits = 0;
  uint8_t ExpcntShift = 4;
  uint8_t LgkmcntShift = 8;
  uint8_t LgkmcntBits = 4;
  uint8_t LoadStorecntShift = 0;
  uint8_t LoadcntBits = 0;
  uint8_t SamplecntBits = 0;
  uint8_t BvhcntBits = 0;
  uint8_t DscntBits = 0;
  uint8_t StorecntBits = 0;
  uint8_t KmcntBits = 0;
  uint8_t XcntBits = 0;
  uint8_t AsynccntBits = 0;
  uint8_t HoldCntBits = 0;
  uint8_t BarrierMemberCountBits = 6;

private:
  auto asTuple() const {
    return std::tie(VmcntLoShift, VmcntLoBits, VmcntHiBits, ExpcntShift,
                    LgkmcntShift, LgkmcntBits, LoadStorecntShift, LoadcntBits,
                    SamplecntBits, BvhcntBits, DscntBits, StorecntBits,
                    KmcntBits, XcntBits, AsynccntBits, HoldCntBits,
                    BarrierMemberCountBits);
  }

public:
  bool operator==(const CounterLayout &O) const {
    return asTuple() == O.asTuple();
  }
};

/// \returns Vmcnt bit mask for \p L.
LLVM_ABI unsigned getVmcntBitMask(const CounterLayout &L);

/// \returns Expcnt bit mask for \p L.
LLVM_ABI unsigned getExpcntBitMask(const CounterLayout &L);

/// \returns Lgkmcnt bit mask for \p L.
LLVM_ABI unsigned getLgkmcntBitMask(const CounterLayout &L);

/// \returns S_WAITCNT immediate mask for \p L.
LLVM_ABI unsigned getWaitcntBitMask(const CounterLayout &L);

/// \returns Loadcnt bit mask for \p L; 0 if absent.
LLVM_ABI unsigned getLoadcntBitMask(const CounterLayout &L);

/// \returns Samplecnt bit mask for \p L; 0 if absent.
LLVM_ABI unsigned getSamplecntBitMask(const CounterLayout &L);

/// \returns Bvhcnt bit mask for \p L; 0 if absent.
LLVM_ABI unsigned getBvhcntBitMask(const CounterLayout &L);

/// \returns Asynccnt bit mask for \p L; 0 if absent.
LLVM_ABI unsigned getAsynccntBitMask(const CounterLayout &L);

/// \returns Dscnt bit mask for \p L; 0 if absent.
LLVM_ABI unsigned getDscntBitMask(const CounterLayout &L);

/// \returns Kmcnt bit mask for \p L; 0 if absent.
LLVM_ABI unsigned getKmcntBitMask(const CounterLayout &L);

/// \returns Xcnt bit mask for \p L; 0 if absent.
LLVM_ABI unsigned getXcntBitMask(const CounterLayout &L);

/// \returns STOREcnt (VScnt on gfx10/gfx11) bit mask for \p L; 0 if absent.
LLVM_ABI unsigned getStorecntBitMask(const CounterLayout &L);

/// \returns Barrier member count bit mask in M0 for \p L.
LLVM_ABI unsigned getBarrierMemberCountBitMask(const CounterLayout &L);

/// \returns Decoded Vmcnt from \p Waitcnt for \p L.
LLVM_ABI unsigned decodeVmcnt(const CounterLayout &L, unsigned Waitcnt);

/// \returns Decoded Expcnt from \p Waitcnt for \p L.
LLVM_ABI unsigned decodeExpcnt(const CounterLayout &L, unsigned Waitcnt);

/// \returns Decoded Lgkmcnt from \p Waitcnt for \p L.
LLVM_ABI unsigned decodeLgkmcnt(const CounterLayout &L, unsigned Waitcnt);

/// \returns Decoded Loadcnt from \p Waitcnt for \p L.
LLVM_ABI unsigned decodeLoadcnt(const CounterLayout &L, unsigned Waitcnt);

/// \returns Decoded Storecnt from \p Waitcnt for \p L.
LLVM_ABI unsigned decodeStorecnt(const CounterLayout &L, unsigned Waitcnt);

/// \returns Decoded Dscnt from \p Waitcnt for \p L.
LLVM_ABI unsigned decodeDscnt(const CounterLayout &L, unsigned Waitcnt);

/// Decodes Vmcnt, Expcnt and Lgkmcnt from S_WAITCNT immediate \p Waitcnt.
/// Should not be used on gfx12+, the instruction which needs it is deprecated.
///
/// \details Layouts:
///     Vmcnt = Waitcnt[3:0]        (pre-gfx9)
///     Vmcnt = Waitcnt[15:14,3:0]  (gfx9,10)
///     Vmcnt = Waitcnt[15:10]      (gfx11)
///     Expcnt = Waitcnt[6:4]       (pre-gfx11)
///     Expcnt = Waitcnt[2:0]       (gfx11)
///     Lgkmcnt = Waitcnt[11:8]     (pre-gfx10)
///     Lgkmcnt = Waitcnt[13:8]     (gfx10)
///     Lgkmcnt = Waitcnt[9:4]      (gfx11)
LLVM_ABI void decodeWaitcnt(const CounterLayout &L, unsigned Waitcnt,
                            unsigned &Vmcnt, unsigned &Expcnt,
                            unsigned &Lgkmcnt);

/// \returns \p Waitcnt with encoded \p Vmcnt for \p L.
LLVM_ABI unsigned encodeVmcnt(const CounterLayout &L, unsigned Waitcnt,
                              unsigned Vmcnt);

/// \returns \p Waitcnt with encoded \p Expcnt for \p L.
LLVM_ABI unsigned encodeExpcnt(const CounterLayout &L, unsigned Waitcnt,
                               unsigned Expcnt);

/// \returns \p Waitcnt with encoded \p Lgkmcnt for \p L.
LLVM_ABI unsigned encodeLgkmcnt(const CounterLayout &L, unsigned Waitcnt,
                                unsigned Lgkmcnt);

/// \returns S_WAITCNT immediate with \p Vmcnt, \p Expcnt and \p Lgkmcnt for
/// \p L. See decodeWaitcnt for the layouts. Should not be used on gfx12+.
LLVM_ABI unsigned encodeWaitcnt(const CounterLayout &L, unsigned Vmcnt,
                                unsigned Expcnt, unsigned Lgkmcnt);

/// \returns S_WAIT_LOADCNT_DSCNT immediate for \p L.
LLVM_ABI unsigned encodeLoadcntDscnt(const CounterLayout &L, unsigned Loadcnt,
                                     unsigned Dscnt);

/// \returns S_WAIT_STORECNT_DSCNT immediate for \p L.
LLVM_ABI unsigned encodeStorecntDscnt(const CounterLayout &L, unsigned Storecnt,
                                      unsigned Dscnt);

namespace DepCtr {

/// \returns Maximum HoldCnt value that can be encoded for \p L.
LLVM_ABI unsigned getHoldCntBitMask(const CounterLayout &L);

/// \returns Decoded HoldCnt from DepCtr immediate \p Encoded for \p L.
LLVM_ABI unsigned decodeFieldHoldCnt(unsigned Encoded, const CounterLayout &L);

/// \returns \p Encoded with \p HoldCnt encoded for \p L.
LLVM_ABI unsigned encodeFieldHoldCnt(unsigned Encoded, unsigned HoldCnt,
                                     const CounterLayout &L);

} // namespace DepCtr

} // namespace llvm::AMDGPU

#endif // LLVM_TARGETPARSER_AMDGPUCOUNTERLAYOUT_H
