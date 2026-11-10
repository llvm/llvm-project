//===--------------- BindCallControllerHandlerSPS.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convenience functions for building call-controller handlers using SPS
// serialization / deserialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_BINDCALLCONTROLLERHANDLERSPS_H
#define LLVM_EXECUTIONENGINE_ORC_BINDCALLCONTROLLERHANDLERSPS_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/ExecutionEngine/Orc/SymbolLookupSet.h"

namespace llvm::orc {

/// Bind a call-controller handler that takes concrete argument types (and a
/// sender for a concrete return type) to the tag with the given name. Uses SPS
/// to unpack the arguments and pack the result.
template <typename SPSSigT, typename HandlerT>
ExecutionSession::CallControllerHandlerBinding bindCallControllerHandlerSPS(
    SymbolNameSpec Name, HandlerT &&Handler,
    SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return ExecutionSession::CallControllerHandlerBinding(
      Name,
      [Handler = std::forward<HandlerT>(Handler)](
          ExecutionSession::CallControllerReturnFn Return,
          shared::WrapperFunctionBuffer ArgBytes) mutable {
        shared::WrapperFunction<SPSSigT>::handleAsync(
            ArgBytes.data(), ArgBytes.size(), std::move(Return), Handler);
      },
      LF);
}

/// Bind a class method as a call-controller handler. The method takes
/// concrete argument types (and a sender for a concrete return type). Uses SPS
/// to unpack the arguments and pack the result.
template <typename SPSSigT, typename ClassT, typename... MethodArgTs>
ExecutionSession::CallControllerHandlerBinding bindCallControllerHandlerSPS(
    SymbolNameSpec Name, ClassT *Instance,
    void (ClassT::*Method)(MethodArgTs...),
    SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return bindCallControllerHandlerSPS<SPSSigT>(
      Name,
      [Instance, Method](MethodArgTs &&...MethodArgs) {
        (Instance->*Method)(std::forward<MethodArgTs>(MethodArgs)...);
      },
      LF);
}

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_BINDCALLCONTROLLERHANDLERSPS_H
