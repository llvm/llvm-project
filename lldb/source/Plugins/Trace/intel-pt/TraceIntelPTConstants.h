//===-- TraceIntelPTConstants.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_CONSTANTS_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_CONSTANTS_H

#include <cstddef>

namespace lldb_private {
namespace trace_intel_pt {

const size_t kThreadBufferSize = 4 * 1024;              // 4KB
const size_t kProcessBufferSizeLimit = 5 * 1024 * 1024; // 500MB

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_CONSTANTS_H
