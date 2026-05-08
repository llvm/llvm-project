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

#define R600_GPU(NAME, ENUM, FEATURES) ENUM,
#define AMDGCN_GPU(NAME, ENUM, ISAVERSION, FEATURES) ENUM,
#include "AMDGPUTargetParser.def"

  GK_AMDGCN_GENERIC_FIRST = GK_GFX9_GENERIC,
  GK_AMDGCN_GENERIC_LAST = GK_GFX12_5_GENERIC,
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
