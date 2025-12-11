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
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    switch (Val) {
    case 0:
      OS << "u8";
      return;
    case 1:
      OS << "u16";
      return;
    case 2:
      OS << "u32";
      return;
    case 3:
      OS << "s32";
      return;
    case 4:
      OS << "u64";
      return;
    case 5:
      OS << "s64";
      return;
    case 6:
      OS << "f16";
      return;
    case 7:
      OS << "f32";
      return;
    case 8:
      OS << "f32.ftz";
      return;
    case 9:
      OS << "f64";
      return;
    case 10:
      OS << "bf16";
      return;
    case 11:
      OS << "tf32";
      return;
    case 12:
      OS << "tf32.ftz";
      return;
    case 13:
      OS << "b4x16";
      return;
    case 14:
      OS << "b4x16_p64";
      return;
    case 15:
      OS << "b6x16_p32";
      return;
    }
  }
  llvm_unreachable("printTensormapElemType called with invalid value for "
                   "immediate argument");
}

void nvvm::printTensormapInterleaveLayout(raw_ostream &OS,
                                          const Constant *ImmArgVal) {
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    switch (Val) {
    case 0:
      OS << "No interleave";
      return;
    case 1:
      OS << "16B interleave";
      return;
    case 2:
      OS << "32B interleave";
      return;
    }
  }
  llvm_unreachable(
      "printTensormapInterleaveLayout called with invalid value for "
      "immediate argument");
}

void nvvm::printTensormapSwizzleMode(raw_ostream &OS,
                                     const Constant *ImmArgVal) {
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    switch (Val) {
    case 0:
      OS << "No swizzling";
      return;
    case 1:
      OS << "32B swizzling";
      return;
    case 2:
      OS << "64B swizzling";
      return;
    case 3:
      OS << "128B swizzling";
      return;
    case 4:
      OS << "96B swizzling";
      return;
    }
  }
  llvm_unreachable("printTensormapSwizzleMode called with invalid value for "
                   "immediate argument");
}

void nvvm::printTensormapSwizzleAtomicity(raw_ostream &OS,
                                          const Constant *ImmArgVal) {
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    switch (Val) {
    case 0:
      OS << "16B";
      return;
    case 1:
      OS << "32B";
      return;
    case 2:
      OS << "32B + 8B flip";
      return;
    case 3:
      OS << "64B";
      return;
    }
  }
  llvm_unreachable(
      "printTensormapSwizzleAtomicity called with invalid value for "
      "immediate argument");
}

void nvvm::printTensormapFillMode(raw_ostream &OS, const Constant *ImmArgVal) {
  if (const auto *CI = dyn_cast<ConstantInt>(ImmArgVal)) {
    uint64_t Val = CI->getZExtValue();
    switch (Val) {
    case 0:
      OS << "Zero fill";
      return;
    case 1:
      OS << "OOB-NaN fill";
      return;
    }
  }
  llvm_unreachable("printTensormapFillMode called with invalid value for "
                   "immediate argument");
}
