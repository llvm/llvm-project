//===- SocketHandle.cpp - Windows SocketHandle stub -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compile-only Windows placeholder. This does not close a native socket and
// must be replaced before SocketHandle is used to own real Windows sockets.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/SocketHandle.h"

namespace orc_rt {

void SocketHandle::reset() noexcept { H = InvalidNativeSocketHandle; }

} // namespace orc_rt
