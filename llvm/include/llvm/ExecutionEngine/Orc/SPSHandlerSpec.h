//===- SPSHandlerSpec.h - SPS call-controller handlers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HandlerSpec binds call-controller handlers (handlers in the controller that
// the ORC runtime calls via its call-controller mechanism) using Simple Packed
// Serialization to decode arguments and encode results.
//
// Each handler's spec derives from HandlerSpec, which supplies the wire
// signature, and adds the default tag name. It plays the same role for calls
// from the executor to the controller that ProxySpec plays for calls from the
// controller to the executor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SPSHANDLERSPEC_H
#define LLVM_EXECUTIONENGINE_ORC_SPSHANDLERSPEC_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"

#include <utility>

namespace llvm::orc::sps {

/// Base class for specs that bind handlers for calls with the SPS signature
/// SPSSigT. Derived specs add the default tag name, e.g.:
///
/// \code{.cpp}
///   struct MyHandlerSpec : sps::HandlerSpec<int32_t(int32_t, int32_t)> {
///     static constexpr SymbolNameSpec Name =
///         SymbolNameSpec::c("my_handler_tag");
///   };
/// \endcode
///
/// Use with bindCallControllerHandler.
template <typename SPSSigT> class HandlerSpec {
public:
  /// The call's SPS signature.
  using SPSSig = SPSSigT;

  /// Wrap H as a handler for the call. H takes a sender for the call's result,
  /// followed by the call's arguments.
  template <typename HandlerT>
  static ExecutionSession::CallControllerHandlerFn wrap(HandlerT &&H) {
    return [H = std::forward<HandlerT>(H)](
               ExecutionSession::CallControllerReturnFn Return,
               shared::WrapperFunctionBuffer ArgBytes) mutable {
      shared::WrapperFunction<SPSSigT>::handleAsync(
          ArgBytes.data(), ArgBytes.size(), std::move(Return), H);
    };
  }
};

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_SPSHANDLERSPEC_H
