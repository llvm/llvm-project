//===-- LibiptDecoder.h --======---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_LIBIPT_DECODER_H
#define LLDB_SOURCE_PLUGINS_TRACE_LIBIPT_DECODER_H

#include "intel-pt.h"

#include "DecodedThread.h"
#include "forward-declarations.h"

namespace lldb_private {
namespace trace_intel_pt {

/// Decode a raw Intel PT trace given in \p buffer and append the decoded
/// instructions and errors in \p decoded_thread. It uses the low level libipt
/// library underneath.
void DecodeTrace(DecodedThread &decoded_thread, TraceIntelPT &trace_intel_pt,
                 llvm::ArrayRef<uint8_t> buffer);

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_LIBIPT_DECODER_H
