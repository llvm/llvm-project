//===- BindCallControllerHandler.h - Bind handlers from specs ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bind call-controller handlers using a handler spec (e.g. sps::HandlerSpec),
// which supplies the default tag name and the serialization protocol. These
// mirror recordProxy, which does the same for proxies.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_BINDCALLCONTROLLERHANDLER_H
#define LLVM_EXECUTIONENGINE_ORC_BINDCALLCONTROLLERHANDLER_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"
#include "llvm/ExecutionEngine/Orc/SymbolLookupSet.h"

#include <utility>

namespace llvm::orc {

/// Bind H, using the given spec, to the tag with the spec's default name.
template <typename HandlerSpecT, typename HandlerT>
ExecutionSession::CallControllerHandlerBinding bindCallControllerHandler(
    HandlerT &&H, SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return ExecutionSession::CallControllerHandlerBinding(
      HandlerSpecT::Name, HandlerSpecT::wrap(std::forward<HandlerT>(H)), LF);
}

/// Bind H, using the given spec, to the tag with the given name rather than
/// the spec's default name.
template <typename HandlerSpecT, typename HandlerT>
ExecutionSession::CallControllerHandlerBinding bindCallControllerHandler(
    SymbolNameSpec Name, HandlerT &&H,
    SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return ExecutionSession::CallControllerHandlerBinding(
      Name, HandlerSpecT::wrap(std::forward<HandlerT>(H)), LF);
}

/// Bind a class method, using the given spec, to the tag with the spec's
/// default name.
template <typename HandlerSpecT, typename ClassT, typename... MethodArgTs>
ExecutionSession::CallControllerHandlerBinding bindCallControllerHandler(
    ClassT *Instance, void (ClassT::*Method)(MethodArgTs...),
    SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return bindCallControllerHandler<HandlerSpecT>(HandlerSpecT::Name, Instance,
                                                 Method, LF);
}

/// Bind a class method, using the given spec, to the tag with the given name
/// rather than the spec's default name.
template <typename HandlerSpecT, typename ClassT, typename... MethodArgTs>
ExecutionSession::CallControllerHandlerBinding bindCallControllerHandler(
    SymbolNameSpec Name, ClassT *Instance,
    void (ClassT::*Method)(MethodArgTs...),
    SymbolLookupFlags LF = SymbolLookupFlags::RequiredSymbol) {
  return bindCallControllerHandler<HandlerSpecT>(
      Name,
      [Instance, Method](MethodArgTs &&...MethodArgs) {
        (Instance->*Method)(std::forward<MethodArgTs>(MethodArgs)...);
      },
      LF);
}

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_BINDCALLCONTROLLERHANDLER_H
