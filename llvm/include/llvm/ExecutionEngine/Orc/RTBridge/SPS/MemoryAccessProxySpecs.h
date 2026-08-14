//===-- MemoryAccessProxySpecs.h - SPS specs for mem access -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS ProxySpecs for the MemoryAccessProxies: each binds a Proxy to its
// controller-interface descriptor in Shared/SPSCI/MemoryAccessSPSCI.h, which
// supplies the wrapper name and wire signature.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_MEMORYACCESSPROXYSPECS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_MEMORYACCESSPROXYSPECS_H

#include "llvm/ExecutionEngine/Orc/RTBridge/MemoryAccessProxies.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/SPS/ProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/MemoryAccessSPSCI.h"

namespace llvm::orc::rt::sps {

using MemWriteUInt8sProxySpec =
    ProxySpec<rt::MemWriteUInt8sProxy, sps_ci::MemWriteUInt8s>;
using MemWriteUInt16sProxySpec =
    ProxySpec<rt::MemWriteUInt16sProxy, sps_ci::MemWriteUInt16s>;
using MemWriteUInt32sProxySpec =
    ProxySpec<rt::MemWriteUInt32sProxy, sps_ci::MemWriteUInt32s>;
using MemWriteUInt64sProxySpec =
    ProxySpec<rt::MemWriteUInt64sProxy, sps_ci::MemWriteUInt64s>;
using MemWritePointersProxySpec =
    ProxySpec<rt::MemWritePointersProxy, sps_ci::MemWritePointers>;
using MemWriteBuffersProxySpec =
    ProxySpec<rt::MemWriteBuffersProxy, sps_ci::MemWriteBuffers>;
using MemReadUInt8sProxySpec =
    ProxySpec<rt::MemReadUInt8sProxy, sps_ci::MemReadUInt8s>;
using MemReadUInt16sProxySpec =
    ProxySpec<rt::MemReadUInt16sProxy, sps_ci::MemReadUInt16s>;
using MemReadUInt32sProxySpec =
    ProxySpec<rt::MemReadUInt32sProxy, sps_ci::MemReadUInt32s>;
using MemReadUInt64sProxySpec =
    ProxySpec<rt::MemReadUInt64sProxy, sps_ci::MemReadUInt64s>;
using MemReadPointersProxySpec =
    ProxySpec<rt::MemReadPointersProxy, sps_ci::MemReadPointers>;
using MemReadBuffersProxySpec =
    ProxySpec<rt::MemReadBuffersProxy, sps_ci::MemReadBuffers>;
using MemReadStringsProxySpec =
    ProxySpec<rt::MemReadStringsProxy, sps_ci::MemReadStrings>;

} // namespace llvm::orc::rt::sps

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_SPS_MEMORYACCESSPROXYSPECS_H
