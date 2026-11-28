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

#ifndef ORC_RT_SUPPORT_ALLOCACTION_H
#define ORC_RT_SUPPORT_ALLOCACTION_H

#include "orc-rt/support/ArgForwardingTraits.h"
#include "orc-rt/support/Error.h"
#include "orc-rt/support/ExecutorAddress.h"
#include "orc-rt/support/WrapperFunction.h"
#include "orc-rt/support/scope_exit.h"

#include <vector>

namespace orc_rt {

/// An AllocActionFn is a function that takes an argument blob and returns an
/// empty WrapperFunctionBuffer on success, or an out-of-band error on failure.
typedef orc_rt_WrapperFunctionBuffer (*AllocActionFn)(const char *ArgData,
                                                      size_t ArgSize) noexcept;

struct AllocActionFunction {

  /// Deserializes allocation action arguments using D, calls H with them,
  /// and serializes H's result using S.
  ///
  /// H may take its arguments by value, const reference, rvalue reference, or
  /// non-const lvalue reference (modifications through the latter are made to
  /// the deserialized value).
  template <typename Deserializer, typename Serializer, typename Handler>
  static WrapperFunctionBuffer handle(const char *ArgData, size_t ArgSize,
                                      Deserializer &&D, Serializer &&S,
                                      Handler &&H) noexcept {
    typedef detail::ArgForwardingTraitsFor<Handler> HandlerTraits;
    typename HandlerTraits::ArgTupleType Args;
    if (!D.deserialize(ArgData, ArgSize, Args))
      return WrapperFunctionBuffer::createOutOfBandError(
          "Could not deserialize allocation action argument buffer");

    return S.serialize(std::apply(
        HandlerTraits::forwardArgsAsRequested(std::forward<Handler>(H)), Args));
  }

  /// Wraps a noexcept method for use as an allocation action handler.
  ///
  /// Method's call operator takes an ExecutorAddr as its first argument,
  /// casts it to a ClassT*, and calls the wrapped method on that object,
  /// forwarding the remaining arguments. The address is not checked: it must
  /// point to a live ClassT instance.
  template <typename ClassT, typename ReturnT, typename... ArgTs>
  struct Method {
    Method(ReturnT (ClassT::*M)(ArgTs...) noexcept) noexcept : M(M) {}
    ReturnT operator()(ExecutorAddr Obj, ArgTs &&...Args) noexcept {
      return (Obj.toPtr<ClassT *>()->*M)(std::forward<ArgTs>(Args)...);
    }

  private:
    ReturnT (ClassT::*M)(ArgTs...) noexcept;
  };

  /// Create a Method wrapper for the given (non-const, noexcept) method
  /// pointer. The action's argument list should start with the address of the
  /// object to call the method on:
  ///
  ///   @code{.cpp}
  ///   class Registrar {
  ///   public:
  ///     Error registerRange(ExecutorAddrRange R) noexcept;
  ///   };
  ///
  ///   ORC_RT_SPS_ALLOC_ACTION_IMPL(
  ///       registrar_register, (SPSExecutorAddr, SPSExecutorAddrRange),
  ///       AllocActionFunction::handleWithMethod(&Registrar::registerRange));
  ///   @endcode
  template <typename ClassT, typename ReturnT, typename... ArgTs>
  static Method<ClassT, ReturnT, ArgTs...>
  handleWithMethod(ReturnT (ClassT::*M)(ArgTs...) noexcept) noexcept {
    return Method<ClassT, ReturnT, ArgTs...>(M);
  }
};

/// An AllocAction is a pair of an AllocActionFn and an argument data buffer.
struct AllocAction {
  AllocAction() = default;
  AllocAction(AllocActionFn Fn, WrapperFunctionBuffer ArgData) noexcept
      : Fn(Fn), ArgData(std::move(ArgData)) {}

  [[nodiscard]] WrapperFunctionBuffer operator()() noexcept {
    assert(Fn && "Attempt to call null action");
    return WrapperFunctionBuffer(Fn(ArgData.data(), ArgData.size()));
  }

  explicit operator bool() const noexcept { return !!Fn; }

  AllocActionFn Fn = nullptr;
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
template <typename ReportErrorFn>
Expected<std::vector<AllocAction>>
runFinalizeActions(std::vector<AllocActionPair> AAPs,
                   ReportErrorFn &&ReportError) noexcept {
  std::vector<AllocAction> DeallocActions;
  auto RunDeallocActions = scope_exit([&]() {
    while (!DeallocActions.empty()) {
      auto B = DeallocActions.back()();
      if (auto *ErrMsg = B.getOutOfBandError())
        ReportError(make_error<StringError>(ErrMsg));
      DeallocActions.pop_back();
    }
  });

  for (auto &AAP : AAPs) {
    if (AAP.Finalize) {
      auto B = AAP.Finalize();
      if (const char *ErrMsg = B.getOutOfBandError())
        return make_error<StringError>(ErrMsg);
    }
    if (AAP.Dealloc)
      DeallocActions.push_back(std::move(AAP.Dealloc));
  }

  RunDeallocActions.release();
  return DeallocActions;
}

/// Run the given deallocation actions in revwerse order.
template <typename ReportErrorFn>
void runDeallocActions(std::vector<AllocAction> DAAs,
                       ReportErrorFn &&ReportError) noexcept {
  while (!DAAs.empty()) {
    auto B = DAAs.back()();
    if (const char *ErrMsg = B.getOutOfBandError())
      ReportError(make_error<StringError>(ErrMsg));
    DAAs.pop_back();
  }
}

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_ALLOCACTION_H
