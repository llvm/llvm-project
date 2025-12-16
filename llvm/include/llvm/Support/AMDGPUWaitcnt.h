//===---------------- AMDGPUWaitcnt.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU waitcnt support infrastructure
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AMDGPUWAITCNT_H
#define LLVM_SUPPORT_AMDGPUWAITCNT_H

#include "llvm/TargetParser/TargetParser.h" // IsaVersion

namespace llvm::AMDGPU {
/// Represents the counter values to wait for in an s_waitcnt instruction.
///
/// Large values (including the maximum possible integer) can be used to
/// represent "don't care" waits.
struct LLVM_ABI Waitcnt {
  unsigned LoadCnt = ~0u; // Corresponds to Vmcnt prior to gfx12.
  unsigned ExpCnt = ~0u;
  unsigned DsCnt = ~0u;     // Corresponds to LGKMcnt prior to gfx12.
  unsigned StoreCnt = ~0u;  // Corresponds to VScnt on gfx10/gfx11.
  unsigned SampleCnt = ~0u; // gfx12+ only.
  unsigned BvhCnt = ~0u;    // gfx12+ only.
  unsigned KmCnt = ~0u;     // gfx12+ only.
  unsigned XCnt = ~0u;      // gfx1250.

  Waitcnt() = default;
  // Pre-gfx12 constructor.
  Waitcnt(unsigned VmCnt, unsigned ExpCnt, unsigned LgkmCnt, unsigned VsCnt)
      : LoadCnt(VmCnt), ExpCnt(ExpCnt), DsCnt(LgkmCnt), StoreCnt(VsCnt) {}

  // gfx12+ constructor.
  Waitcnt(unsigned LoadCnt, unsigned ExpCnt, unsigned DsCnt, unsigned StoreCnt,
          unsigned SampleCnt, unsigned BvhCnt, unsigned KmCnt, unsigned XCnt)
      : LoadCnt(LoadCnt), ExpCnt(ExpCnt), DsCnt(DsCnt), StoreCnt(StoreCnt),
        SampleCnt(SampleCnt), BvhCnt(BvhCnt), KmCnt(KmCnt), XCnt(XCnt) {}

  bool hasWait() const { return StoreCnt != ~0u || hasWaitExceptStoreCnt(); }

  bool hasWaitExceptStoreCnt() const {
    return LoadCnt != ~0u || ExpCnt != ~0u || DsCnt != ~0u ||
           SampleCnt != ~0u || BvhCnt != ~0u || KmCnt != ~0u || XCnt != ~0u;
  }

  bool hasWaitStoreCnt() const { return StoreCnt != ~0u; }

  Waitcnt combined(const Waitcnt &Other) const {
    // Does the right thing provided self and Other are either both pre-gfx12
    // or both gfx12+.
    return Waitcnt(
        std::min(LoadCnt, Other.LoadCnt), std::min(ExpCnt, Other.ExpCnt),
        std::min(DsCnt, Other.DsCnt), std::min(StoreCnt, Other.StoreCnt),
        std::min(SampleCnt, Other.SampleCnt), std::min(BvhCnt, Other.BvhCnt),
        std::min(KmCnt, Other.KmCnt), std::min(XCnt, Other.XCnt));
  }

  friend raw_ostream &operator<<(raw_ostream &OS, const AMDGPU::Waitcnt &Wait);
};

// The following methods are only meaningful on targets that support
// S_WAITCNT.

/// \returns Vmcnt bit mask for given isa \p Version.
LLVM_ABI unsigned getVmcntBitMask(const IsaVersion &Version);

/// \returns Expcnt bit mask for given isa \p Version.
LLVM_ABI unsigned getExpcntBitMask(const IsaVersion &Version);

/// \returns Lgkmcnt bit mask for given isa \p Version.
LLVM_ABI unsigned getLgkmcntBitMask(const IsaVersion &Version);

/// \returns Waitcnt bit mask for given isa \p Version.
LLVM_ABI unsigned getWaitcntBitMask(const IsaVersion &Version);

/// \returns Decoded Vmcnt from given \p Waitcnt for given isa \p Version.
LLVM_ABI unsigned decodeVmcnt(const IsaVersion &Version, unsigned Waitcnt);

/// \returns Decoded Expcnt from given \p Waitcnt for given isa \p Version.
LLVM_ABI unsigned decodeExpcnt(const IsaVersion &Version, unsigned Waitcnt);

/// \returns Decoded Lgkmcnt from given \p Waitcnt for given isa \p Version.
LLVM_ABI unsigned decodeLgkmcnt(const IsaVersion &Version, unsigned Waitcnt);

/// Decodes Vmcnt, Expcnt and Lgkmcnt from given \p Waitcnt for given isa
/// \p Version, and writes decoded values into \p Vmcnt, \p Expcnt and
/// \p Lgkmcnt respectively. Should not be used on gfx12+, the instruction
/// which needs it is deprecated
///
/// \details \p Vmcnt, \p Expcnt and \p Lgkmcnt are decoded as follows:
///     \p Vmcnt = \p Waitcnt[3:0]        (pre-gfx9)
///     \p Vmcnt = \p Waitcnt[15:14,3:0]  (gfx9,10)
///     \p Vmcnt = \p Waitcnt[15:10]      (gfx11)
///     \p Expcnt = \p Waitcnt[6:4]       (pre-gfx11)
///     \p Expcnt = \p Waitcnt[2:0]       (gfx11)
///     \p Lgkmcnt = \p Waitcnt[11:8]     (pre-gfx10)
///     \p Lgkmcnt = \p Waitcnt[13:8]     (gfx10)
///     \p Lgkmcnt = \p Waitcnt[9:4]      (gfx11)
///
LLVM_ABI void decodeWaitcnt(const IsaVersion &Version, unsigned Waitcnt,
                            unsigned &Vmcnt, unsigned &Expcnt,
                            unsigned &Lgkmcnt);

LLVM_ABI Waitcnt decodeWaitcnt(const IsaVersion &Version, unsigned Encoded);

/// \returns \p Waitcnt with encoded \p Vmcnt for given isa \p Version.
LLVM_ABI unsigned encodeVmcnt(const IsaVersion &Version, unsigned Waitcnt,
                              unsigned Vmcnt);

/// \returns \p Waitcnt with encoded \p Expcnt for given isa \p Version.
LLVM_ABI unsigned encodeExpcnt(const IsaVersion &Version, unsigned Waitcnt,
                               unsigned Expcnt);

/// \returns \p Waitcnt with encoded \p Lgkmcnt for given isa \p Version.
LLVM_ABI unsigned encodeLgkmcnt(const IsaVersion &Version, unsigned Waitcnt,
                                unsigned Lgkmcnt);

/// Encodes \p Vmcnt, \p Expcnt and \p Lgkmcnt into Waitcnt for given isa
/// \p Version. Should not be used on gfx12+, the instruction which needs
/// it is deprecated
///
/// \details \p Vmcnt, \p Expcnt and \p Lgkmcnt are encoded as follows:
///     Waitcnt[2:0]   = \p Expcnt      (gfx11+)
///     Waitcnt[3:0]   = \p Vmcnt       (pre-gfx9)
///     Waitcnt[3:0]   = \p Vmcnt[3:0]  (gfx9,10)
///     Waitcnt[6:4]   = \p Expcnt      (pre-gfx11)
///     Waitcnt[9:4]   = \p Lgkmcnt     (gfx11)
///     Waitcnt[11:8]  = \p Lgkmcnt     (pre-gfx10)
///     Waitcnt[13:8]  = \p Lgkmcnt     (gfx10)
///     Waitcnt[15:10] = \p Vmcnt       (gfx11)
///     Waitcnt[15:14] = \p Vmcnt[5:4]  (gfx9,10)
///
/// \returns Waitcnt with encoded \p Vmcnt, \p Expcnt and \p Lgkmcnt for given
/// isa \p Version.
///
LLVM_ABI unsigned encodeWaitcnt(const IsaVersion &Version, unsigned Vmcnt,
                                unsigned Expcnt, unsigned Lgkmcnt);

LLVM_ABI unsigned encodeWaitcnt(const IsaVersion &Version,
                                const Waitcnt &Decoded);

// The following methods are only meaningful on targets that support
// S_WAIT_*CNT, introduced with gfx12.

/// \returns Loadcnt bit mask for given isa \p Version.
/// Returns 0 for versions that do not support LOADcnt
LLVM_ABI unsigned getLoadcntBitMask(const IsaVersion &Version);

/// \returns Samplecnt bit mask for given isa \p Version.
/// Returns 0 for versions that do not support SAMPLEcnt
LLVM_ABI unsigned getSamplecntBitMask(const IsaVersion &Version);

/// \returns Bvhcnt bit mask for given isa \p Version.
/// Returns 0 for versions that do not support BVHcnt
LLVM_ABI unsigned getBvhcntBitMask(const IsaVersion &Version);

/// \returns Dscnt bit mask for given isa \p Version.
/// Returns 0 for versions that do not support DScnt
LLVM_ABI unsigned getDscntBitMask(const IsaVersion &Version);

/// \returns Dscnt bit mask for given isa \p Version.
/// Returns 0 for versions that do not support KMcnt
LLVM_ABI unsigned getKmcntBitMask(const IsaVersion &Version);

/// \returns Xcnt bit mask for given isa \p Version.
/// Returns 0 for versions that do not support Xcnt.
LLVM_ABI unsigned getXcntBitMask(const IsaVersion &Version);

/// \return STOREcnt or VScnt bit mask for given isa \p Version.
/// returns 0 for versions that do not support STOREcnt or VScnt.
/// STOREcnt and VScnt are the same counter, the name used
/// depends on the ISA version.
LLVM_ABI unsigned getStorecntBitMask(const IsaVersion &Version);

// The following are only meaningful on targets that support
// S_WAIT_LOADCNT_DSCNT and S_WAIT_STORECNT_DSCNT.

/// \returns Decoded Waitcnt structure from given \p LoadcntDscnt for given
/// isa \p Version.
LLVM_ABI Waitcnt decodeLoadcntDscnt(const IsaVersion &Version,
                                    unsigned LoadcntDscnt);

/// \returns Decoded Waitcnt structure from given \p StorecntDscnt for given
/// isa \p Version.
LLVM_ABI Waitcnt decodeStorecntDscnt(const IsaVersion &Version,
                                     unsigned StorecntDscnt);

/// \returns \p Loadcnt and \p Dscnt components of \p Decoded  encoded as an
/// immediate that can be used with S_WAIT_LOADCNT_DSCNT for given isa
/// \p Version.
LLVM_ABI unsigned encodeLoadcntDscnt(const IsaVersion &Version,
                                     const Waitcnt &Decoded);

/// \returns \p Storecnt and \p Dscnt components of \p Decoded  encoded as an
/// immediate that can be used with S_WAIT_STORECNT_DSCNT for given isa
/// \p Version.
LLVM_ABI unsigned encodeStorecntDscnt(const IsaVersion &Version,
                                      const Waitcnt &Decoded);
} // end namespace llvm::AMDGPU

#endif // LLVM_SUPPORT_AMDGPUWAITCNT_H
