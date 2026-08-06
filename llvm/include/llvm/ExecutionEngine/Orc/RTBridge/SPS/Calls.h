//===------------- Calls.h - SPS-based Call Wrappers ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS-based implementations of the RTBridge caller interfaces.
//
// These implement the rt::Caller interfaces by invoking executor-side wrapper
// functions in the runtime's controller interface, using Simple Packed
// Serialization to encode arguments and decode results.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLS_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/Calls.h"

namespace llvm::orc::rt::sps {

template <typename CallerT, typename SPSSigT, const char *DefaultName,
          typename FnType = typename CallerT::FnType>
class CallerSpec;

template <typename CallerT, typename SPSSigT, const char *DefaultName,
          typename RetT, typename... ArgTs>
class CallerSpec<CallerT, SPSSigT, DefaultName, RetT(ArgTs...)> {
  using CalleeRetT = typename CallerT::CalleeRetT;
  using ErrorRetT = typename CallerT::ErrorRetT;

public:
  static constexpr const char *Name = DefaultName;

  static void dispatch(unique_function<void(ErrorRetT)> OnComplete,
                       ExecutionSession &ES, ExecutorAddr CalleeAddr,
                       const ArgTs &...Args) {
    if constexpr (std::is_void_v<CalleeRetT>) {
      // Void result: the executor-side function produces no value, so the only
      // thing to report is the dispatch error (success if the call ran).
      ES.callSPSWrapperAsync<SPSSigT>(CalleeAddr, std::move(OnComplete),
                                      Args...);
    } else {
      ES.callSPSWrapperAsync<SPSSigT>(
          CalleeAddr,
          [OnComplete = std::move(OnComplete)](Error SerErr,
                                               CalleeRetT Result) mutable {
            if (SerErr)
              return OnComplete(std::move(SerErr));
            return OnComplete(std::move(Result));
          },
          Args...);
    }
  }
};

using CallMainSPSSig = int64_t(shared::SPSExecutorAddr,
                               shared::SPSSequence<shared::SPSString>);
inline constexpr char CallMainCIName[] = "orc_rt_ci_sps_call_main";
/// SPS caller for rt::MainCaller: runs a main-like function
/// (int(int argc, char *argv[])) in the executor.
using MainCallerSpec =
    CallerSpec<rt::MainCaller, CallMainSPSSig, CallMainCIName>;

using CallVoidVoidSPSSig = void(shared::SPSExecutorAddr);
inline constexpr char CallVoidVoidCIName[] = "orc_rt_ci_sps_call_void_void";
/// SPS caller for rt::VoidVoidCaller: runs a void() function in the executor.
/// WARNING: This Caller is experimental and may be removed.
using VoidVoidCallerSpec =
    CallerSpec<rt::VoidVoidCaller, CallVoidVoidSPSSig, CallVoidVoidCIName>;

using CallInt32VoidSPSSig = int32_t(shared::SPSExecutorAddr);
inline constexpr char CallInt32VoidCIName[] = "orc_rt_ci_sps_call_int32_void";
/// SPS caller for rt::Int32VoidCaller: runs an int32_t() function in the
/// executor.
/// WARNING: This Caller is experimental and may be removed.
using Int32VoidCallerSpec =
    CallerSpec<rt::Int32VoidCaller, CallInt32VoidSPSSig, CallInt32VoidCIName>;

using CallInt32Int32SPSSig = int32_t(shared::SPSExecutorAddr, int32_t);
inline constexpr char CallInt32Int32CIName[] = "orc_rt_ci_sps_call_int32_int32";
/// SPS caller for rt::Int32Int32Caller: runs an int32_t(int32_t) function in
/// the executor.
/// WARNING: This Caller is experimental and may be removed.
using Int32Int32CallerSpec =
    CallerSpec<rt::Int32Int32Caller, CallInt32Int32SPSSig,
               CallInt32Int32CIName>;

} // namespace llvm::orc::rt::sps

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLS_H
