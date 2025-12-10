//===- AllocActionTestUtils.h ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_ALLOCACTIONTESTUTILS_H
#define ORC_RT_UNITTEST_ALLOCACTIONTESTUTILS_H

#include "SimplePackedSerializationTestUtils.h"
#include "orc-rt/AllocAction.h"

#include <optional>

template <typename... SPSArgTs> struct MakeAllocAction {
  template <typename... ArgTs>
  static std::optional<orc_rt::AllocAction> from(orc_rt::AllocActionFn Fn,
                                                 ArgTs &&...Args) {
    using SPS = orc_rt::SPSArgList<SPSArgTs...>;
    auto B = orc_rt::WrapperFunctionBuffer::allocate(SPS::size(Args...));
    orc_rt::SPSOutputBuffer OB(B.data(), B.size());
    if (!SPS::serialize(OB, Args...))
      return std::nullopt;
    return orc_rt::AllocAction(Fn, std::move(B));
  }
};

#endif // ORC_RT_UNITTEST_ALLOCACTIONTESTUTILS_H
