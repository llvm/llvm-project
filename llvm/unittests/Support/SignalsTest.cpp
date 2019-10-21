//========- unittests/Support/SignalsTest.cpp - Signal handling test =========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if !defined(_WIN32)
#include <unistd.h>
#include <sysexits.h>
#include <signal.h>
#endif // !defined(_WIN32)

#include "llvm/Support/Signals.h"

#include "gtest/gtest.h"

using namespace llvm;

#if !defined(_WIN32)
TEST(SignalTest, IgnoreMultipleSIGPIPEs) {
  // Ignore SIGPIPE.
  signal(SIGPIPE, SIG_IGN);

  // Disable exit-on-SIGPIPE.
  sys::SetPipeSignalFunction(nullptr);

  // Create unidirectional read/write pipes.
  int fds[2];
  int err = pipe(fds);
  if (err != 0)
    return; // If we can't make pipes, this isn't testing anything.

  // Close the read pipe.
  close(fds[0]);

  // Attempt to write to the write pipe. Currently we're asserting that the
  // write fails, which isn't great.
  //
  // What we really want is a death test that checks that this block exits
  // with a special exit "success" code, as opposed to unexpectedly exiting due
  // to a kill-by-SIGNAL or due to the default SIGPIPE handler.
  //
  // Unfortunately llvm's unit tests aren't set up to support death tests well.
  // For one, death tests are flaky in a multithreaded context. And sigactions
  // inherited from llvm-lit interfere with what's being tested.
  const void *buf = (const void *)&fds;
  err = write(fds[1], buf, 1);
  ASSERT_EQ(err, -1);
  err = write(fds[1], buf, 1);
  ASSERT_EQ(err, -1);
}
#endif // !defined(_WIN32)
