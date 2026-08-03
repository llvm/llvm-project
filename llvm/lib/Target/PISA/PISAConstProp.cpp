//===-- PISAConstProp.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAConstProp.h"
#include "llvm/IR/Constants.h"

#include <cmath>

namespace llvm {
namespace PISA {
namespace ConstProp {

Constant *foldFrcp(ConstantFP *C0) {
  auto APF = C0->getValueAPF();
  double C0value = C0->getType()->isFloatTy()
                       ? static_cast<double>(APF.convertToFloat())
                       : APF.convertToDouble();
  if (C0->isNaN())
    return ConstantFP::getQNaN(C0->getType(), C0->isNegative());
  if (!C0value)
    return ConstantFP::getInfinity(C0->getType(), C0->isNegative());

  return ConstantFP::get(C0->getType(), 1. / C0value);
}

Constant *foldFrsqrt(ConstantFP *C0) {
  auto APF = C0->getValueAPF();
  double C0value = C0->getType()->isFloatTy()
                       ? static_cast<double>(APF.convertToFloat())
                       : APF.convertToDouble();
  if (C0->isNaN() || C0value < 0)
    return ConstantFP::getQNaN(C0->getType(), C0->isNegative());
  if (!C0value)
    return ConstantFP::getInfinity(C0->getType());

  return ConstantFP::get(C0->getType(), sqrt(1. / C0value));
}

Constant *foldFtanh(ConstantFP *C0) {
  auto APF = C0->getValueAPF();
  double C0value = C0->getType()->isFloatTy()
                       ? static_cast<double>(APF.convertToFloat())
                       : APF.convertToDouble();
  if (C0->isNaN())
    return ConstantFP::getQNaN(C0->getType(), C0->isNegative());

  const double Th = tanh(C0value);
  return ConstantFP::get(C0->getType(), Th);
}

} // namespace ConstProp
} // namespace PISA
} // namespace llvm
