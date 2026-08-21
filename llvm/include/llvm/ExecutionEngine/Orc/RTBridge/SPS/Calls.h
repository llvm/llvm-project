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
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"

namespace llvm::orc::rt::sps {

/// Implements the rt::Caller interface BaseT by calling an executor-side SPS
/// wrapper function, using SPSSigT to encode the arguments and decode the
/// result.
///
/// The wrapper is a controller-interface (CI) entry point named CIName: a
/// wrapper function (byte blob in, byte blob out) that the runtime exposes to
/// the controller. SPSSigT is the Simple Packed Serialization signature used to
/// encode the argument blob and decode the result blob; it must be compatible
/// with BaseT's FnType. CIName must have static storage duration (e.g. an
/// inline constexpr char[]).
///
/// The FnType parameter is deduced from BaseT and should not be supplied
/// explicitly; the primary template is left undefined so that only the
/// RetT(ArgTs...) specialization can be instantiated.
template <typename BaseT, typename SPSSigT, const char *CINameV,
          typename FnType = typename BaseT::FnType>
class Caller;

template <typename BaseT, typename SPSSigT, const char *CINameV, typename RetT,
          typename... ArgTs>
class Caller<BaseT, SPSSigT, CINameV, RetT(ArgTs...)> : public BaseT {
  using CalleeRetT = typename BaseT::CalleeRetT;
  using ErrorRetT = typename BaseT::ErrorRetT;

public:
  /// Name of the controller-interface wrapper this caller targets.
  static constexpr const char *CIName = CINameV;

  using BaseT::BaseT;
  using BaseT::operator();

  /// Look the wrapper up in the executor's bootstrap JITDylib and build a
  /// caller for it.
  static Expected<Caller>
  Create(ExecutionSession &ES,
         SymbolLookupFlags SLF = SymbolLookupFlags::RequiredSymbol,
         const char *Name = CIName) {
    if (auto CalleeSyms =
            ES.lookup(makeJITDylibSearchOrder(&ES.getBootstrapJITDylib()),
                      SymbolLookupSet{ES.intern(Name), SLF})) {
      if (!CalleeSyms->empty())
        return Caller(ES, CalleeSyms->begin()->second.getAddress());
      assert(SLF == SymbolLookupFlags::WeaklyReferencedSymbol);
      return Caller(ES, ExecutorAddr());
    } else
      return CalleeSyms.takeError();
  }

  /// Asynchronously call the SPS wrapper at CalleeAddr with the given Args,
  /// delivering the result (or an error) to OnComplete. Serialization failures
  /// are reported through OnComplete's error channel.
  static void callAsync(unique_function<void(ErrorRetT)> OnComplete,
                        ExecutionSession &ES, ExecutorAddr CalleeAddr,
                        const ArgTs &...Args) {
    using namespace llvm::orc::shared;
    if constexpr (std::is_void_v<CalleeRetT>) {
      // Void result: the executor-side function produces no value, so the only
      // thing to report is the dispatch error (success if the call ran).
      ES.callSPSWrapperAsync<SPSSigT>(
          CalleeAddr,
          [OnComplete = std::move(OnComplete)](Error SerErr) mutable {
            OnComplete(std::move(SerErr));
          },
          Args...);
    } else {
      ES.callSPSWrapperAsync<SPSSigT>(
          CalleeAddr,
          [OnComplete = std::move(OnComplete)](Error SerErr,
                                               CalleeRetT Result) mutable {
            if (SerErr)
              return OnComplete(std::move(SerErr));
            else
              return OnComplete(std::move(Result));
          },
          Args...);
    }
  }

  void operator()(unique_function<void(ErrorRetT)> OnComplete,
                  ArgTs... Args) override {
    callAsync(std::move(OnComplete), this->executionSession(),
              this->calleeAddr(), Args...);
  }
};

using CallMainSPSSig = int64_t(shared::SPSExecutorAddr,
                               shared::SPSSequence<shared::SPSString>);
inline constexpr char CallMainCIName[] = "orc_rt_ci_sps_call_main";
/// SPS caller for rt::MainCaller: runs a main-like function
/// (int(int argc, char *argv[])) in the executor.
using MainCaller = Caller<rt::MainCaller, CallMainSPSSig, CallMainCIName>;

using CallVoidVoidSPSSig = void(shared::SPSExecutorAddr);
inline constexpr char CallVoidVoidCIName[] = "orc_rt_ci_sps_call_void_void";
/// SPS caller for rt::VoidVoidCaller: runs a void() function in the executor.
/// WARNING: This Caller is experimental and may be removed.
using VoidVoidCaller =
    Caller<rt::VoidVoidCaller, CallVoidVoidSPSSig, CallVoidVoidCIName>;

using CallInt32VoidSPSSig = int32_t(shared::SPSExecutorAddr);
inline constexpr char CallInt32VoidCIName[] = "orc_rt_ci_sps_call_int32_void";
/// SPS caller for rt::Int32VoidCaller: runs an int32_t() function in the
/// executor.
/// WARNING: This Caller is experimental and may be removed.
using Int32VoidCaller =
    Caller<rt::Int32VoidCaller, CallInt32VoidSPSSig, CallInt32VoidCIName>;

using CallInt32Int32SPSSig = int32_t(shared::SPSExecutorAddr, int32_t);
inline constexpr char CallInt32Int32CIName[] = "orc_rt_ci_sps_call_int32_int32";
/// SPS caller for rt::Int32Int32Caller: runs an int32_t(int32_t) function in
/// the executor.
/// WARNING: This Caller is experimental and may be removed.
using Int32Int32Caller =
    Caller<rt::Int32Int32Caller, CallInt32Int32SPSSig, CallInt32Int32CIName>;

} // namespace llvm::orc::rt::sps

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_CALLS_H
