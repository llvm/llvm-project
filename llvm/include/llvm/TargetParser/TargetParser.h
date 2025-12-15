//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_TARGETPARSER_H
#define LLVM_TARGETPARSER_TARGETPARSER_H

#include "SubtargetFeature.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

template <typename T> class SmallVectorImpl;
class Triple;

// Target specific information in their own namespaces.
// (ARM/AArch64/X86 are declared in ARM/AArch64/X86TargetParser.h)
// These should be generated from TableGen because the information is already
// there, and there is where new information about targets will be added.
// FIXME: To TableGen this we need to make some table generated files available
// even if the back-end is not compiled with LLVM, plus we need to create a new
// back-end to TableGen to create these clean tables.
namespace AMDGPU {

/// GPU kinds supported by the AMDGPU target.
enum GPUKind : uint32_t {
  // Not specified processor.
  GK_NONE = 0,

  // R600-based processors.
  GK_R600,
  GK_R630,
  GK_RS880,
  GK_RV670,
  GK_RV710,
  GK_RV730,
  GK_RV770,
  GK_CEDAR,
  GK_CYPRESS,
  GK_JUNIPER,
  GK_REDWOOD,
  GK_SUMO,
  GK_BARTS,
  GK_CAICOS,
  GK_CAYMAN,
  GK_TURKS,

  GK_R600_FIRST = GK_R600,
  GK_R600_LAST = GK_TURKS,

  // AMDGCN-based processors.
  GK_GFX600,
  GK_GFX601,
  GK_GFX602,

  GK_GFX700,
  GK_GFX701,
  GK_GFX702,
  GK_GFX703,
  GK_GFX704,
  GK_GFX705,

  GK_GFX801,
  GK_GFX802,
  GK_GFX803,
  GK_GFX805,
  GK_GFX810,

  GK_GFX900,
  GK_GFX902,
  GK_GFX904,
  GK_GFX906,
  GK_GFX908,
  GK_GFX909,
  GK_GFX90A,
  GK_GFX90C,
  GK_GFX942,
  GK_GFX950,

  GK_GFX1010,
  GK_GFX1011,
  GK_GFX1012,
  GK_GFX1013,
  GK_GFX1030,
  GK_GFX1031,
  GK_GFX1032,
  GK_GFX1033,
  GK_GFX1034,
  GK_GFX1035,
  GK_GFX1036,

  GK_GFX1100,
  GK_GFX1101,
  GK_GFX1102,
  GK_GFX1103,
  GK_GFX1150,
  GK_GFX1151,
  GK_GFX1152,
  GK_GFX1153,

  GK_GFX1200,
  GK_GFX1201,
  GK_GFX1250,
  GK_GFX1251,

  GK_AMDGCN_FIRST = GK_GFX600,
  GK_AMDGCN_LAST = GK_GFX1251,

  GK_GFX9_GENERIC,
  GK_GFX10_1_GENERIC,
  GK_GFX10_3_GENERIC,
  GK_GFX11_GENERIC,
  GK_GFX12_GENERIC,
  GK_GFX9_4_GENERIC,

  GK_AMDGCN_GENERIC_FIRST = GK_GFX9_GENERIC,
  GK_AMDGCN_GENERIC_LAST = GK_GFX9_4_GENERIC,
};

/// Instruction set architecture version.
struct IsaVersion {
  unsigned Major;
  unsigned Minor;
  unsigned Stepping;
};

// This isn't comprehensive for now, just things that are needed from the
// frontend driver.
enum ArchFeatureKind : uint32_t {
  FEATURE_NONE = 0,

  // These features only exist for r600, and are implied true for amdgcn.
  FEATURE_FMA = 1 << 1,
  FEATURE_LDEXP = 1 << 2,
  FEATURE_FP64 = 1 << 3,

  // Common features.
  FEATURE_FAST_FMA_F32 = 1 << 4,
  FEATURE_FAST_DENORMAL_F32 = 1 << 5,

  // Wavefront 32 is available.
  FEATURE_WAVE32 = 1 << 6,

  // Xnack is available.
  FEATURE_XNACK = 1 << 7,

  // Sram-ecc is available.
  FEATURE_SRAMECC = 1 << 8,

  // WGP mode is supported.
  FEATURE_WGP = 1 << 9,

  // Xnack is available by default
  FEATURE_XNACK_ALWAYS = 1 << 10
};

enum FeatureError : uint32_t {
  NO_ERROR = 0,
  INVALID_FEATURE_COMBINATION,
  UNSUPPORTED_TARGET_FEATURE
};

LLVM_ABI StringRef getArchFamilyNameAMDGCN(GPUKind AK);

LLVM_ABI StringRef getArchNameAMDGCN(GPUKind AK);
LLVM_ABI StringRef getArchNameR600(GPUKind AK);
LLVM_ABI StringRef getCanonicalArchName(const Triple &T, StringRef Arch);
LLVM_ABI GPUKind parseArchAMDGCN(StringRef CPU);
LLVM_ABI GPUKind parseArchR600(StringRef CPU);
LLVM_ABI unsigned getArchAttrAMDGCN(GPUKind AK);
LLVM_ABI unsigned getArchAttrR600(GPUKind AK);

LLVM_ABI void fillValidArchListAMDGCN(SmallVectorImpl<StringRef> &Values);
LLVM_ABI void fillValidArchListR600(SmallVectorImpl<StringRef> &Values);

LLVM_ABI IsaVersion getIsaVersion(StringRef GPU);

/// Fills Features map with default values for given target GPU.
/// \p Features contains overriding target features and this function returns
/// default target features with entries overridden by \p Features.
LLVM_ABI std::pair<FeatureError, StringRef>
fillAMDGPUFeatureMap(StringRef GPU, const Triple &T, StringMap<bool> &Features);

/// Represents the counter values to wait for in an s_waitcnt instruction.
///
/// Large values (including the maximum possible integer) can be used to
/// represent "don't care" waits.
struct Waitcnt {
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
} // namespace AMDGPU

struct BasicSubtargetFeatureKV {
  const char *Key;         ///< K-V key string
  unsigned Value;          ///< K-V integer value
  FeatureBitArray Implies; ///< K-V bit mask
};

/// Used to provide key value pairs for feature and CPU bit flags.
struct BasicSubtargetSubTypeKV {
  const char *Key;         ///< K-V key string
  FeatureBitArray Implies; ///< K-V bit mask

  /// Compare routine for std::lower_bound
  bool operator<(StringRef S) const { return StringRef(Key) < S; }

  /// Compare routine for std::is_sorted.
  bool operator<(const BasicSubtargetSubTypeKV &Other) const {
    return StringRef(Key) < StringRef(Other.Key);
  }
};

LLVM_ABI std::optional<llvm::StringMap<bool>>
getCPUDefaultTargetFeatures(StringRef CPU,
                            ArrayRef<BasicSubtargetSubTypeKV> ProcDesc,
                            ArrayRef<BasicSubtargetFeatureKV> ProcFeatures);
} // namespace llvm

#endif
