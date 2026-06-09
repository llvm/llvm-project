//===- CallSPSCI.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface registration for function callers.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/sps-ci/CallSPSCI.h"
#include "orc-rt/SPSWrapperFunction.h"
#include "orc-rt/move_only_function.h"

#include <string>
#include <vector>

namespace orc_rt::sps_ci {

using VoidVoidFn = void();

void call_void_void(move_only_function<void()> Return, VoidVoidFn Fn) {
  Fn();
  Return();
}

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_call_void_void, void(SPSExecutorAddr),
                   call_void_void);

using MainFn = int(int argc, char *argv[]);

static std::vector<char *>
makeNullTerminatedCStringArray(std::vector<std::string> &U) {
  std::vector<char *> V;
  V.reserve(U.size() + 1);
  for (auto &E : U)
    V.push_back(E.data());
  V.push_back(nullptr);
  return V;
}

void call_main(move_only_function<void(int64_t)> Return, MainFn Main,
               std::vector<std::string> Args) {
  auto ArgV = makeNullTerminatedCStringArray(Args);
  Return(Main(ArgV.size(), ArgV.data()));
}

ORC_RT_SPS_WRAPPER(orc_rt_ci_sps_call_main,
                   int64_t(SPSExecutorAddr, SPSSequence<SPSString>), call_main);

static std::pair<const char *, const void *> orc_rt_ci_sps_call_interface[] = {
    ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_call_void_void),
    ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_call_main)};

Error addCall(SimpleSymbolTable &ST) {
  return ST.addUnique(orc_rt_ci_sps_call_interface);
}

} // namespace orc_rt::sps_ci
