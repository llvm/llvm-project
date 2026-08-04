//===------------- Calls.h - SPS-based Call Wrappers ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS-based wrappers for calling functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLS_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/Calls.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"

namespace llvm::orc::rt::sps {

/// Calls the executor-side "call-main" wrapper, which runs a program's main
/// function with the given arguments and returns its result.
class MainCaller : public rt::MainCaller {
public:
  /// Name of the controller interface wrapper function.
  static constexpr const char *CIName = "orc_rt_ci_sps_call_main";

  static void callAsync(unique_function<void(Expected<int64_t>)> OnComplete,
                        ExecutionSession &ES, ExecutorAddr CallMainFnAddr,
                        ExecutorAddr MainFnAddr, ArrayRef<std::string> Args) {
    using namespace llvm::orc::shared;
    ES.callSPSWrapperAsync<int64_t(SPSExecutorAddr, SPSSequence<SPSString>)>(
        CallMainFnAddr,
        [OnComplete = std::move(OnComplete)](Error SerErr,
                                             int64_t Result) mutable {
          if (SerErr)
            return OnComplete(std::move(SerErr));
          else
            return OnComplete(Result);
        },
        MainFnAddr, Args);
  }

  MainCaller(ExecutionSession &ES, ExecutorAddr CallMainFnAddr)
      : ES(ES), CallMainFnAddr(CallMainFnAddr) {}

  /// Look up the call-main wrapper in the executor's bootstrap JITDylib and
  /// build a MainCaller for it.
  static Expected<MainCaller> Create(ExecutionSession &ES) {
    if (auto CallMainSym = ES.lookup({&ES.getBootstrapJITDylib()}, CIName))
      return MainCaller(ES, CallMainSym->getAddress());
    else
      return CallMainSym.takeError();
  }

  void operator()(unique_function<void(Expected<int64_t>)> OnComplete,
                  ExecutorAddr MainFnAddr,
                  ArrayRef<std::string> Args) override {
    callAsync(std::move(OnComplete), ES, CallMainFnAddr, MainFnAddr, Args);
  }

  using rt::MainCaller::operator();

private:
  ExecutionSession &ES;
  ExecutorAddr CallMainFnAddr;
};

} // namespace llvm::orc::rt::sps

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLS_H
