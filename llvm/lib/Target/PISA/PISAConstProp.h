//===-- PISAConstProp.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISACONSTPROP_H
#define LLVM_LIB_TARGET_PISA_PISACONSTPROP_H

#include "llvm/IR/Constants.h"

namespace llvm {
namespace PISA {
///
/// Namespace implementing constant folding/propagation (PISA own additional
/// functionality to common LLVM constant folding/propagation).
///
namespace ConstProp {
// Frcp (1/x)
Constant *foldFrcp(ConstantFP *C0);
// Frsqrt (1/sqrt(x))
Constant *foldFrsqrt(ConstantFP *C0);
// Hyperbolic tangent
Constant *foldFtanh(ConstantFP *C0);
} // namespace ConstProp
} // namespace PISA
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISACONSTPROP_H
