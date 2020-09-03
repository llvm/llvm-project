//===-- CommunicationTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Communication.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/Pipe.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

#ifndef _WIN32
TEST(CommunicationTest, SynchronizeWhileClosing) {
  // Set up a communication object reading from a pipe.
  Pipe pipe;
  ASSERT_THAT_ERROR(pipe.CreateNew(/*child_process_inherit=*/false).ToError(),
                    llvm::Succeeded());

  Communication comm("test");
  comm.SetConnection(std::make_unique<ConnectionFileDescriptor>(
      pipe.ReleaseReadFileDescriptor(), /*owns_fd=*/true));
  comm.SetCloseOnEOF(true);
  ASSERT_TRUE(comm.StartReadThread());

  // Ensure that we can safely synchronize with the read thread while it is
  // closing the read end (in response to us closing the write end).
  pipe.CloseWriteFileDescriptor();
  comm.SynchronizeWithReadThread();

  ASSERT_TRUE(comm.StopReadThread());
}
#endif
