//===- AllSPSCI.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of sps_ci::addAll.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/sps-ci/AllSPSCI.h"

namespace orc_rt::sps_ci {

Error addAll(SimpleSymbolTable &ST) {
  using AdderFn = Error (*)(SimpleSymbolTable &);
  AdderFn Adders[] = {addSimpleNativeMemoryMap};

  for (auto *Adder : Adders)
    if (auto Err = Adder(ST))
      return Err;

  return Error::success();
}

} // namespace orc_rt::sps_ci
