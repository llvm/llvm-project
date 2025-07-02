//==-- LoongArch64TargetParser - Parser for LoongArch64 features --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise LoongArch hardware features
// such as CPU/ARCH and extension names.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_LOONGARCHTARGETPARSER_H
#define LLVM_TARGETPARSER_LOONGARCHTARGETPARSER_H

#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"
#include <vector>

namespace llvm {
class StringRef;

namespace LoongArch {

enum FeatureKind : uint32_t {
  // 64-bit ISA is available.
  FK_64BIT = 1 << 1,

  // Single-precision floating-point instructions are available.
  FK_FP32 = 1 << 2,

  // Double-precision floating-point instructions are available.
  FK_FP64 = 1 << 3,

  // Loongson SIMD Extension is available.
  FK_LSX = 1 << 4,

  // Loongson Advanced SIMD Extension is available.
  FK_LASX = 1 << 5,

  // Loongson Binary Translation Extension is available.
  FK_LBT = 1 << 6,

  // Loongson Virtualization Extension is available.
  FK_LVZ = 1 << 7,

  // Allow memory accesses to be unaligned.
  FK_UAL = 1 << 8,

  // Floating-point approximate reciprocal instructions are available.
  FK_FRECIPE = 1 << 9,

  // Atomic memory swap and add instructions for byte and half word are
  // available.
  FK_LAM_BH = 1 << 10,

  // Atomic memory compare and swap instructions for byte, half word, word and
  // double word are available.
  FK_LAMCAS = 1 << 11,

  // Do not generate load-load barrier instructions (dbar 0x700).
  FK_LD_SEQ_SA = 1 << 12,

  // Assume div.w[u] and mod.w[u] can handle inputs that are not sign-extended.
  FK_DIV32 = 1 << 13,

  // sc.q is available.
  FK_SCQ = 1 << 14,
};

struct FeatureInfo {
  StringRef Name;
  FeatureKind Kind;
};

enum class ArchKind {
#define LOONGARCH_ARCH(NAME, KIND, FEATURES) KIND,
#include "LoongArchTargetParser.def"
};

struct ArchInfo {
  StringRef Name;
  ArchKind Kind;
  uint32_t Features;
};

LLVM_ABI bool isValidArchName(StringRef Arch);
LLVM_ABI bool isValidFeatureName(StringRef Feature);
LLVM_ABI bool getArchFeatures(StringRef Arch, std::vector<StringRef> &Features);
LLVM_ABI bool isValidCPUName(StringRef TuneCPU);
LLVM_ABI void fillValidCPUList(SmallVectorImpl<StringRef> &Values);
LLVM_ABI StringRef getDefaultArch(bool Is64Bit);

} // namespace LoongArch

} // namespace llvm

#endif // LLVM_TARGETPARSER_LOONGARCHTARGETPARSER_H
