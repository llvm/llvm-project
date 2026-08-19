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
#include "orc-rt/MacroUtils.h"
#include "orc-rt/SPSWrapperFunctionBuffer.h"
#include "orc-rt/SimplePackedSerialization.h"

/// Define an allocation-action wrapper function with the given Name that
/// uses SPS to deserialize its arguments and dispatches to Handle.
///
/// SPSArgs is a parenthesized comma-separated list of SPS argument types
/// (the parens are stripped by ORC_RT_DEPAREN before being expanded into
/// the SPSAllocActionFunction template instantiation):
///
///     static Error checkEq(int32_t X, int32_t Y);
///     ORC_RT_SPS_ALLOC_ACTION(check_eq_action, (int32_t, int32_t), checkEq)
///
#define ORC_RT_SPS_ALLOC_ACTION(Name, SPSArgs, Handle)                         \
  static orc_rt_WrapperFunctionBuffer Name(const char *ArgData,                \
                                           size_t ArgSize) {                   \
    return orc_rt::SPSAllocActionFunction<ORC_RT_DEPAREN(SPSArgs)>::handle(    \
               ArgData, ArgSize, Handle)                                       \
        .release();                                                            \
  }

namespace orc_rt {

struct SPSAllocAction;

template <> class SPSSerializationTraits<SPSAllocAction, AllocAction> {
public:
  static size_t size(const AllocAction &AA) {
    return SPSArgList<SPSExecutorAddr, SPSWrapperFunctionBuffer>::size(
        ExecutorAddr::fromPtr(AA.Fn), AA.ArgData);
  }

  static bool serialize(SPSOutputBuffer &OB, const AllocAction &AA) {
    return SPSArgList<SPSExecutorAddr, SPSWrapperFunctionBuffer>::serialize(
        OB, ExecutorAddr::fromPtr(AA.Fn), AA.ArgData);
  }

  static bool deserialize(SPSInputBuffer &IB, AllocAction &AA) {
    ExecutorAddr Fn;
    WrapperFunctionBuffer ArgData;
    if (!SPSArgList<SPSExecutorAddr, SPSWrapperFunctionBuffer>::deserialize(
            IB, Fn, ArgData))
      return false;
    AA.Fn = Fn.toPtr<AllocActionFn>();
    AA.ArgData = std::move(ArgData);
    return true;
  }
};

struct SPSAllocActionPair;

template <> class SPSSerializationTraits<SPSAllocActionPair, AllocActionPair> {
public:
  static size_t size(const AllocActionPair &AAP) {
    return SPSArgList<SPSAllocAction, SPSAllocAction>::size(AAP.Finalize,
                                                            AAP.Dealloc);
  }

  static bool serialize(SPSOutputBuffer &OB, const AllocActionPair &AAP) {
    return SPSArgList<SPSAllocAction, SPSAllocAction>::serialize(
        OB, AAP.Finalize, AAP.Dealloc);
  }

  static bool deserialize(SPSInputBuffer &IB, AllocActionPair &AAP) {
    return SPSArgList<SPSAllocAction, SPSAllocAction>::deserialize(
        IB, AAP.Finalize, AAP.Dealloc);
  }
};

struct AllocActionSPSSerializer {

  /// Pass-through for handlers returning WrapperFunctionBuffer.
  static WrapperFunctionBuffer serialize(WrapperFunctionBuffer B) { return B; }

  /// Error serialization:
  ///   - success values converted to empty WrapperFunctionBuffers
  ///   - failure values converted to out-of-band errors.
  static WrapperFunctionBuffer serialize(Error Err) {
    if (!Err)
      return WrapperFunctionBuffer();
    return WrapperFunctionBuffer::createOutOfBandError(
        toString(std::move(Err)).c_str());
  }
};

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
        AllocActionSPSSerializer(), std::forward<Handler>(H));
  }
};

} // namespace orc_rt

#endif // ORC_RT_SPSALLOCACTION_H
