//===--- MemoryAccessProxies.h - Proxies for memory access ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Named rt::Proxy types for the executor's memory-access operations. Unlike the
// Call proxies, these target wrappers that perform the operation directly, so
// they take the operation's data arguments rather than a callee address.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_MEMORYACCESSPROXIES_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_MEMORYACCESSPROXIES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/Proxy.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llvm::orc::rt {

using MemWriteUInt8sProxy = Proxy<void(ArrayRef<tpctypes::UInt8Write>)>;
using MemWriteUInt16sProxy = Proxy<void(ArrayRef<tpctypes::UInt16Write>)>;
using MemWriteUInt32sProxy = Proxy<void(ArrayRef<tpctypes::UInt32Write>)>;
using MemWriteUInt64sProxy = Proxy<void(ArrayRef<tpctypes::UInt64Write>)>;
using MemWritePointersProxy = Proxy<void(ArrayRef<tpctypes::PointerWrite>)>;
using MemWriteBuffersProxy = Proxy<void(ArrayRef<tpctypes::BufferWrite>)>;
using MemReadUInt8sProxy = Proxy<std::vector<uint8_t>(ArrayRef<ExecutorAddr>)>;
using MemReadUInt16sProxy =
    Proxy<std::vector<uint16_t>(ArrayRef<ExecutorAddr>)>;
using MemReadUInt32sProxy =
    Proxy<std::vector<uint32_t>(ArrayRef<ExecutorAddr>)>;
using MemReadUInt64sProxy =
    Proxy<std::vector<uint64_t>(ArrayRef<ExecutorAddr>)>;
using MemReadPointersProxy =
    Proxy<std::vector<ExecutorAddr>(ArrayRef<ExecutorAddr>)>;
using MemReadBuffersProxy =
    Proxy<std::vector<std::vector<uint8_t>>(ArrayRef<ExecutorAddrRange>)>;
using MemReadStringsProxy =
    Proxy<std::vector<std::string>(ArrayRef<ExecutorAddr>)>;

} // namespace llvm::orc::rt

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_MEMORYACCESSPROXIES_H
