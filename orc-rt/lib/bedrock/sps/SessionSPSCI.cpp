//===- SessionSPSCI.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface implementation for Session APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/SessionSPSCI.h"

#include "orc-rt/bedrock/Session.h"
#include "orc-rt/support/move_only_function.h"
#include "orc-rt/support/sps/SPSSymbolLookupSet.h"

// Wrapper for Session methods that uses the wrapper's Session argument.
#define ORC_RT_SPS_SESSION_WRAPPER_IMPL(Name, SPSSig, HandlerBuilder)          \
  ORC_RT_SPS_WRAPPER_DECL(Name)                                                \
  extern "C" ORC_RT_SPS_WRAPPER_SIG(Name) {                                    \
    orc_rt::SPSWrapperFunction<SPSSig>::handle(S, ArgBytes, Return, CallId,    \
                                               (HandlerBuilder)(S));           \
  }

namespace orc_rt::sps_ci {

namespace {

template <typename RetT, typename... ArgTs> class SessionSyncMethod {
public:
  SessionSyncMethod(RetT (Session::*M)(ArgTs...) const noexcept) noexcept
      : M(M) {}

  // Not a handler, but a handler-builder: This is the HandlerBuilder function
  // used in the macro above -- it takes a Session and returns a wrapper to be
  // used in the WrapperFunction::handle method.
  auto operator()(orc_rt_SessionRef S) noexcept {
    assert(S && "Session pointer must not be null");
    return [S, M = M](move_only_function<void(RetT)> Return,
                      ArgTs &&...Args) noexcept {
      Return((unwrap(S)->*M)(std::forward<ArgTs>(Args)...));
    };
  }

private:
  RetT (Session::*M)(ArgTs...) const noexcept;
};

template <typename RetT, typename... ArgTs>
SessionSyncMethod(RetT (Session::*M)(ArgTs...) const noexcept)
    -> SessionSyncMethod<RetT, ArgTs...>;

} // namespace

ORC_RT_SPS_SESSION_WRAPPER_IMPL(
    orc_rt_ci_sps_Session_lookupInstanceSymbols,
    SPSSymbolLookupResult(SPSSymbolLookupSet),
    SessionSyncMethod(&Session::lookupInstanceSymbols))

static std::pair<SymbolNameSpec, const void *>
    orc_rt_ci_Session_sps_interface[] = {
        ORC_RT_SYMTAB_C_PAIR(orc_rt_ci_sps_Session_lookupInstanceSymbols)};

Error addSession(SimpleSymbolTable &ST) noexcept {
  return ST.addUnique(orc_rt_ci_Session_sps_interface);
}

} // namespace orc_rt::sps_ci
