//===- SocketHandle.cpp - POSIX SocketHandle implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// POSIX implementation of orc-rt/bedrock/SocketHandle.h.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/SocketHandle.h"

#include "orc-rt-c/support/Logging.h"
#include "orc-rt-internal/support/sys/Errno.h"

#include <cerrno>
#include <string>
#include <unistd.h>

namespace orc_rt {

void SocketHandle::reset() noexcept {
  if (H == InvalidNativeSocketHandle)
    return;
  // Not retried on EINTR: close releases the descriptor before the steps that
  // can fail, so a retry could close one that another thread has since been
  // given.
  if (::close(H) != 0 && errno != EINTR) {
    [[maybe_unused]] std::string Msg = sys::strError(errno);
    ORC_RT_LOG(Info, General, "SocketHandle: close failed: %s", Msg.c_str());
  }
  H = InvalidNativeSocketHandle;
}

} // namespace orc_rt
