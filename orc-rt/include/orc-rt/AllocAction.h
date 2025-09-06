//===---------- AllocAction.h - Allocation action APIs ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AllocAction and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_ALLOCACTION_H
#define ORC_RT_ALLOCACTION_H

#include "orc-rt/Error.h"
#include "orc-rt/WrapperFunction.h"

#include <vector>

namespace orc_rt {
namespace detail {

template <typename Handler>
struct AAHandlerTraits
    : public AAHandlerTraits<
          decltype(&std::remove_cv_t<std::remove_reference_t<Handler>>::
                   operator())> {};

template <typename... ArgTs>
struct AAHandlerTraits<WrapperFunctionBuffer(ArgTs...)> {
  typedef std::tuple<ArgTs...> ArgTuple;
};

template <typename Class, typename... ArgTs>
struct AAHandlerTraits<WrapperFunctionBuffer (Class::*)(ArgTs...)>
    : public AAHandlerTraits<WrapperFunctionBuffer(ArgTs...)> {};

template <typename Class, typename... ArgTs>
struct AAHandlerTraits<WrapperFunctionBuffer (Class::*)(ArgTs...) const>
    : public AAHandlerTraits<WrapperFunctionBuffer(ArgTs...)> {};

} // namespace detail

/// An AllocActionFn is a function that takes an argument blob and returns an
/// empty WrapperFunctionBuffer on success, or an out-of-band error on failure.
typedef orc_rt_WrapperFunctionBuffer (*AllocActionFn)(const char *ArgData,
                                                      size_t ArgSize);

struct AllocActionFunction {

  template <typename Deserializer, typename Handler>
  static WrapperFunctionBuffer handle(const char *ArgData, size_t ArgSize,
                                      Deserializer &&D, Handler &&H) {
    typename detail::AAHandlerTraits<Handler>::ArgTuple Args;
    if (!D.deserialize(ArgData, ArgSize, Args))
      return WrapperFunctionBuffer::createOutOfBandError(
          "Could not deserialize allocation action argument buffer");

    return std::apply(std::forward<Handler>(H), std::move(Args)).release();
  }
};

/// An AllocAction is a pair of an AllocActionFn and an argument data buffer.
struct AllocAction {
  AllocAction() = default;
  AllocAction(AllocActionFn AA, WrapperFunctionBuffer ArgData)
      : AA(AA), ArgData(std::move(ArgData)) {}

  [[nodiscard]] WrapperFunctionBuffer operator()() {
    assert(AA && "Attempt to call null action");
    return AA(ArgData.data(), ArgData.size());
  }

  explicit operator bool() const noexcept { return !!AA; }

  AllocActionFn AA = nullptr;
  WrapperFunctionBuffer ArgData;
};

/// An AllocActionPair is a pair of a Finalize action and a Dealloc action.
struct AllocActionPair {
  AllocAction Finalize;
  AllocAction Dealloc;
};

/// Run the finalize actions in the given sequence.
///
/// On success, returns the list of deallocation actions to be run in reverse
/// order at deallocation time.
///
/// On failure, runs deallocation actions associated with any previously
/// successful finalize actions, then returns an error.
///
/// Both finalize and dealloc actions are permitted to be null (i.e. have a
/// null action function) in which case they are ignored.
[[nodiscard]] Expected<std::vector<AllocAction>>
runFinalizeActions(std::vector<AllocActionPair> AAPs);

/// Run the given deallocation actions in revwerse order.
void runDeallocActions(std::vector<AllocAction> DAAs);

} // namespace orc_rt

#endif // ORC_RT_ALLOCACTION_H
