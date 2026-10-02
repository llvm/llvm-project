//===- SocketTestUtils.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sockets for tests of the socket APIs, defined once per system under
// sys/<system>/ so that a test needing one stays portable.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_BEDROCK_SOCKETTESTUTILS_H
#define ORC_RT_UNITTEST_BEDROCK_SOCKETTESTUTILS_H

#include "orc-rt/bedrock/SocketHandle.h"

#include <optional>

/// Creates a socket for a test to own, or nullopt if the system refuses one.
/// The socket is neither bound nor connected.
std::optional<orc_rt::NativeSocketHandle> makeNativeSocket();

/// True if H names a socket this process still has open.
///
/// Only meaningful while nothing else in the process is opening sockets: a
/// closed handle's value can be reissued to the next caller, which is
/// indistinguishable from the original still being open.
bool isNativeSocketOpen(orc_rt::NativeSocketHandle H);

/// Closes H, which must be open and owned by no SocketHandle.
void closeNativeSocket(orc_rt::NativeSocketHandle H);

#endif // ORC_RT_UNITTEST_BEDROCK_SOCKETTESTUTILS_H
