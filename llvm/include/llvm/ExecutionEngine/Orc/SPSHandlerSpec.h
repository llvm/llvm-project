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
// A HandlerSpec is built from a handler descriptor, which supplies the default
// tag name and the wire signature. It mirrors ProxySpec, which plays the same
// role for calls from the controller to the executor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SPSHANDLERSPEC_H
#define LLVM_EXECUTIONENGINE_ORC_SPSHANDLERSPEC_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"

#include <utility>

namespace llvm::orc::sps {

/// Binds handlers for the call described by Descriptor, which must provide:
///
///   static constexpr SymbolNameSpec Name; // The default tag name.
///   using SPSSig = ...;                   // The call's SPS signature.
///
/// Use with bindCallControllerHandler.
template <typename Descriptor> class HandlerSpec {
public:
  static constexpr SymbolNameSpec Name = Descriptor::Name;

  /// Wrap H as a handler for the call. H takes a sender for the call's result,
  /// followed by the call's arguments.
  template <typename HandlerT>
  static ExecutionSession::CallControllerHandlerFn wrap(HandlerT &&H) {
    return [H = std::forward<HandlerT>(H)](
               ExecutionSession::CallControllerReturnFn Return,
               shared::WrapperFunctionBuffer ArgBytes) mutable {
      shared::WrapperFunction<typename Descriptor::SPSSig>::handleAsync(
          ArgBytes.data(), ArgBytes.size(), std::move(Return), H);
    };
  }
};

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_SPSHANDLERSPEC_H
