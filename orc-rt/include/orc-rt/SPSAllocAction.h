//===---- SPSAllocAction.h - SPS-serialized AllocAction utils ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for implementing allocation actions that take an SPS-serialized
// argument buffer.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SPSALLOCACTION_H
#define ORC_RT_SPSALLOCACTION_H

#include "orc-rt/AllocAction.h"
#include "orc-rt/SimplePackedSerialization.h"

namespace orc_rt {

template <typename... SPSArgTs> struct AllocActionSPSDeserializer {
  template <typename... ArgTs>
  bool deserialize(const char *ArgData, size_t ArgSize, ArgTs &...Args) {
    SPSInputBuffer IB(ArgData, ArgSize);
    return SPSArgList<SPSArgTs...>::deserialize(IB, Args...);
  }
};

/// Provides call and handle utilities to simplify writing and invocation of
/// wrapper functions that use SimplePackedSerialization to serialize and
/// deserialize their arguments and return values.
template <typename... SPSArgTs> struct SPSAllocActionFunction {

  template <typename Handler>
  static WrapperFunctionBuffer handle(const char *ArgData, size_t ArgSize,
                                      Handler &&H) {
    return AllocActionFunction::handle(
        ArgData, ArgSize, AllocActionSPSDeserializer<SPSTuple<SPSArgTs...>>(),
        std::forward<Handler>(H));
  }
};

} // namespace orc_rt

#endif // ORC_RT_SPSALLOCACTION_H
