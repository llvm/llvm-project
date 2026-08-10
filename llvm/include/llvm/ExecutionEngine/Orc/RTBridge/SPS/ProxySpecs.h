//===-------- ProxySpecs.h - SPS-based Call Wrappers ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS-based implementations of the RTBridge proxy interfaces.
//
// These implement the rt::Proxy interfaces by invoking executor-side wrapper
// functions in the runtime's controller interface, using Simple Packed
// Serialization to encode arguments and decode results.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_PROXYSPECS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_PROXYSPECS_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/Proxy.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>

namespace llvm::orc::rt::sps {

template <typename ProxyT, typename SPSSigT, const char *DefaultName,
          typename FnType = typename ProxyT::FnType>
class ProxySpec;

template <typename ProxyT, typename SPSSigT, const char *DefaultName,
          typename RetT, typename... ArgTs>
class ProxySpec<ProxyT, SPSSigT, DefaultName, RetT(ArgTs...)> {
  using CalleeRetT = typename ProxyT::CalleeRetT;
  using ErrorRetT = typename ProxyT::ErrorRetT;

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
/// SPS proxy for rt::CallMainProxy: runs a main-like function
/// (int(int argc, char *argv[])) in the executor.
using CallMainProxySpec =
    ProxySpec<rt::CallMainProxy, CallMainSPSSig, CallMainCIName>;

using CallVoidVoidSPSSig = void(shared::SPSExecutorAddr);
inline constexpr char CallVoidVoidCIName[] = "orc_rt_ci_sps_call_void_void";
/// SPS proxy for rt::CallVoidVoidProxy: runs a void() function in the executor.
/// WARNING: This Proxy is experimental and may be removed.
using CallVoidVoidProxySpec =
    ProxySpec<rt::CallVoidVoidProxy, CallVoidVoidSPSSig, CallVoidVoidCIName>;

using CallInt32VoidSPSSig = int32_t(shared::SPSExecutorAddr);
inline constexpr char CallInt32VoidCIName[] = "orc_rt_ci_sps_call_int32_void";
/// SPS proxy for rt::CallInt32VoidProxy: runs an int32_t() function in the
/// executor.
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32VoidProxySpec =
    ProxySpec<rt::CallInt32VoidProxy, CallInt32VoidSPSSig, CallInt32VoidCIName>;

using CallInt32Int32SPSSig = int32_t(shared::SPSExecutorAddr, int32_t);
inline constexpr char CallInt32Int32CIName[] = "orc_rt_ci_sps_call_int32_int32";
/// SPS proxy for rt::CallInt32Int32Proxy: runs an int32_t(int32_t) function in
/// the executor.
/// WARNING: This Proxy is experimental and may be removed.
using CallInt32Int32ProxySpec =
    ProxySpec<rt::CallInt32Int32Proxy, CallInt32Int32SPSSig,
              CallInt32Int32CIName>;

// Memory-access proxies. Unlike the Call* proxies above, these target wrappers
// that perform the operation directly, so they take the operation's data
// arguments and no callee address.

using MemWriteUInt8sSPSSig =
    void(shared::SPSSequence<shared::SPSMemoryAccessUInt8Write>);
inline constexpr char MemWriteUInt8sCIName[] = "orc_rt_ci_sps_mem_write_uint8s";
using MemWriteUInt8sProxySpec =
    ProxySpec<rt::MemWriteUInt8sProxy, MemWriteUInt8sSPSSig,
              MemWriteUInt8sCIName>;

using MemWriteUInt16sSPSSig =
    void(shared::SPSSequence<shared::SPSMemoryAccessUInt16Write>);
inline constexpr char MemWriteUInt16sCIName[] =
    "orc_rt_ci_sps_mem_write_uint16s";
using MemWriteUInt16sProxySpec =
    ProxySpec<rt::MemWriteUInt16sProxy, MemWriteUInt16sSPSSig,
              MemWriteUInt16sCIName>;

using MemWriteUInt32sSPSSig =
    void(shared::SPSSequence<shared::SPSMemoryAccessUInt32Write>);
inline constexpr char MemWriteUInt32sCIName[] =
    "orc_rt_ci_sps_mem_write_uint32s";
using MemWriteUInt32sProxySpec =
    ProxySpec<rt::MemWriteUInt32sProxy, MemWriteUInt32sSPSSig,
              MemWriteUInt32sCIName>;

using MemWriteUInt64sSPSSig =
    void(shared::SPSSequence<shared::SPSMemoryAccessUInt64Write>);
inline constexpr char MemWriteUInt64sCIName[] =
    "orc_rt_ci_sps_mem_write_uint64s";
using MemWriteUInt64sProxySpec =
    ProxySpec<rt::MemWriteUInt64sProxy, MemWriteUInt64sSPSSig,
              MemWriteUInt64sCIName>;

using MemWritePointersSPSSig =
    void(shared::SPSSequence<shared::SPSMemoryAccessPointerWrite>);
inline constexpr char MemWritePointersCIName[] =
    "orc_rt_ci_sps_mem_write_pointers";
using MemWritePointersProxySpec =
    ProxySpec<rt::MemWritePointersProxy, MemWritePointersSPSSig,
              MemWritePointersCIName>;

using MemWriteBuffersSPSSig =
    void(shared::SPSSequence<shared::SPSMemoryAccessBufferWrite>);
inline constexpr char MemWriteBuffersCIName[] =
    "orc_rt_ci_sps_mem_write_buffers";
using MemWriteBuffersProxySpec =
    ProxySpec<rt::MemWriteBuffersProxy, MemWriteBuffersSPSSig,
              MemWriteBuffersCIName>;

using MemReadUInt8sSPSSig =
    shared::SPSSequence<uint8_t>(shared::SPSSequence<shared::SPSExecutorAddr>);
inline constexpr char MemReadUInt8sCIName[] = "orc_rt_ci_sps_mem_read_uint8s";
using MemReadUInt8sProxySpec =
    ProxySpec<rt::MemReadUInt8sProxy, MemReadUInt8sSPSSig, MemReadUInt8sCIName>;

using MemReadUInt16sSPSSig =
    shared::SPSSequence<uint16_t>(shared::SPSSequence<shared::SPSExecutorAddr>);
inline constexpr char MemReadUInt16sCIName[] = "orc_rt_ci_sps_mem_read_uint16s";
using MemReadUInt16sProxySpec =
    ProxySpec<rt::MemReadUInt16sProxy, MemReadUInt16sSPSSig,
              MemReadUInt16sCIName>;

using MemReadUInt32sSPSSig =
    shared::SPSSequence<uint32_t>(shared::SPSSequence<shared::SPSExecutorAddr>);
inline constexpr char MemReadUInt32sCIName[] = "orc_rt_ci_sps_mem_read_uint32s";
using MemReadUInt32sProxySpec =
    ProxySpec<rt::MemReadUInt32sProxy, MemReadUInt32sSPSSig,
              MemReadUInt32sCIName>;

using MemReadUInt64sSPSSig =
    shared::SPSSequence<uint64_t>(shared::SPSSequence<shared::SPSExecutorAddr>);
inline constexpr char MemReadUInt64sCIName[] = "orc_rt_ci_sps_mem_read_uint64s";
using MemReadUInt64sProxySpec =
    ProxySpec<rt::MemReadUInt64sProxy, MemReadUInt64sSPSSig,
              MemReadUInt64sCIName>;

using MemReadPointersSPSSig = shared::SPSSequence<shared::SPSExecutorAddr>(
    shared::SPSSequence<shared::SPSExecutorAddr>);
inline constexpr char MemReadPointersCIName[] =
    "orc_rt_ci_sps_mem_read_pointers";
using MemReadPointersProxySpec =
    ProxySpec<rt::MemReadPointersProxy, MemReadPointersSPSSig,
              MemReadPointersCIName>;

using MemReadBuffersSPSSig = shared::SPSSequence<shared::SPSSequence<uint8_t>>(
    shared::SPSSequence<shared::SPSExecutorAddrRange>);
inline constexpr char MemReadBuffersCIName[] = "orc_rt_ci_sps_mem_read_buffers";
using MemReadBuffersProxySpec =
    ProxySpec<rt::MemReadBuffersProxy, MemReadBuffersSPSSig,
              MemReadBuffersCIName>;

using MemReadStringsSPSSig = shared::SPSSequence<shared::SPSString>(
    shared::SPSSequence<shared::SPSExecutorAddr>);
inline constexpr char MemReadStringsCIName[] = "orc_rt_ci_sps_mem_read_strings";
using MemReadStringsProxySpec =
    ProxySpec<rt::MemReadStringsProxy, MemReadStringsSPSSig,
              MemReadStringsCIName>;

} // namespace llvm::orc::rt::sps

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_PROXYSPECS_H
