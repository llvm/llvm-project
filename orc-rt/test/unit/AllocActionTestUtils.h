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
#include "orc-rt/support/AllocAction.h"

#include <optional>

namespace orc_rt::test {

template <typename... SPSArgTs> struct MakeAllocAction {
  template <typename... ArgTs>
  static std::optional<AllocAction> from(AllocActionFn Fn, ArgTs &&...Args) {
    using SPS = SPSArgList<SPSArgTs...>;
    auto B = WrapperFunctionBuffer::allocate(SPS::size(Args...));
    SPSOutputBuffer OB(B.data(), B.size());
    if (!SPS::serialize(OB, Args...))
      return std::nullopt;
    return AllocAction(Fn, std::move(B));
  }
};

} // namespace orc_rt::test

#endif // ORC_RT_UNITTEST_ALLOCACTIONTESTUTILS_H
