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
