//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions associated with NVVM Intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/NVVMIntrinsicUtils.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace nvvm;

void nvvm::printTcgen05MMAKind(raw_ostream &OS, const Constant *ImmArgVal) {
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    switch (static_cast<Tcgen05MMAKind>(Val)) {
    case Tcgen05MMAKind::F16:
      OS << "f16";
      return;
    case Tcgen05MMAKind::TF32:
      OS << "tf32";
      return;
    case Tcgen05MMAKind::F8F6F4:
      OS << "f8f6f4";
      return;
    case Tcgen05MMAKind::I8:
      OS << "i8";
      return;
    }
  }
  llvm_unreachable(
      "printTcgen05MMAKind called with invalid value for immediate argument");
}

void nvvm::printTcgen05CollectorUsageOp(raw_ostream &OS,
                                        const Constant *ImmArgVal) {
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    switch (static_cast<Tcgen05CollectorUsageOp>(Val)) {
    case Tcgen05CollectorUsageOp::DISCARD:
      OS << "discard";
      return;
    case Tcgen05CollectorUsageOp::LASTUSE:
      OS << "lastuse";
      return;
    case Tcgen05CollectorUsageOp::FILL:
      OS << "fill";
      return;
    case Tcgen05CollectorUsageOp::USE:
      OS << "use";
      return;
    }
  }
  llvm_unreachable("printTcgen05CollectorUsageOp called with invalid value for "
                   "immediate argument");
}

void nvvm::printTensormapElemType(raw_ostream &OS, const Constant *ImmArgVal) {
  static constexpr StringRef TensormapElemTypes[] = {
      "u8",       "u16",   "u32",       "s32",      "u64",  "s64",
      "f16",      "f32",   "f32.ftz",   "f64",      "bf16", "tf32",
      "tf32.ftz", "b4x16", "b4x16_p64", "b6x16_p32"};
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    if (Val <= static_cast<uint64_t>(nvvm::TensormapElemType::B6x16_p32)) {
      OS << TensormapElemTypes[Val];
      return;
    }
  }
}

void nvvm::printTensormapInterleaveLayout(raw_ostream &OS,
                                          const Constant *ImmArgVal) {
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    switch (static_cast<TensormapInterleaveLayout>(Val)) {
    case TensormapInterleaveLayout::NO_INTERLEAVE:
      OS << "No interleave";
      return;
    case TensormapInterleaveLayout::INTERLEAVE_16B:
      OS << "16B interleave";
      return;
    case TensormapInterleaveLayout::INTERLEAVE_32B:
      OS << "32B interleave";
      return;
    }
  }
}

void nvvm::printTensormapSwizzleMode(raw_ostream &OS,
                                     const Constant *ImmArgVal) {
  static constexpr StringRef TensormapSwizzleModes[] = {
      "No swizzling", "32B swizzling", "64B swizzling", "128B swizzling",
      "96B swizzling"};
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    if (Val <= static_cast<uint64_t>(nvvm::TensormapSwizzleMode::SWIZZLE_96B)) {
      OS << TensormapSwizzleModes[Val];
      return;
    }
  }
}

void nvvm::printTensormapSwizzleAtomicity(raw_ostream &OS,
                                          const Constant *ImmArgVal) {
  static constexpr StringRef TensormapSwizzleAtomicities[] = {
      "16B", "32B", "32B + 8B flip", "64B"};
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    if (Val <= static_cast<uint64_t>(
                   nvvm::TensormapSwizzleAtomicity::SWIZZLE_ATOMICITY_64B)) {
      OS << TensormapSwizzleAtomicities[Val];
      return;
    }
  }
}

void nvvm::printTensormapFillMode(raw_ostream &OS, const Constant *ImmArgVal) {
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    OS << (Val == static_cast<uint64_t>(TensormapFillMode::ZERO_FILL)
               ? "Zero fill"
               : "OOB-NaN fill");
    return;
  }
}
