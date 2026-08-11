//===-- MemoryAccessProxySpecs.h - SPS specs for mem access -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS ProxySpecs (signatures, controller-interface names, and dispatch) for the
// MemoryAccessProxies.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_MEMORYACCESSPROXYSPECS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_MEMORYACCESSPROXYSPECS_H

#include "llvm/ExecutionEngine/Orc/RTBridge/MemoryAccessProxies.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/SPS/ProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>

namespace llvm::orc::rt::sps {

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

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_MEMORYACCESSPROXYSPECS_H
