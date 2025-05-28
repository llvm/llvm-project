//===-- ProtocolUtils.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/ProtocolUtils.h"

namespace lldb_dap::protocol {

bool IsAssemblySource(const protocol::Source &source) {
  return source.sourceReference.value_or(0) != 0;
}

} // namespace lldb_dap::protocol
