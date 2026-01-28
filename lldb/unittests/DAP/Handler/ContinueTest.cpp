//===-- ContinueTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Handler/RequestHandler.h"
#include "Protocol/ProtocolRequests.h"
#include "TestBase.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap_tests;
using namespace lldb_dap::protocol;

class ContinueRequestHandlerTest : public DAPTestBase {};

TEST_F(ContinueRequestHandlerTest, NotStopped) {
  SBTarget target;
  dap->debugger.SetSelectedTarget(target);

  ContinueRequestHandler handler(*dap);

  ContinueArguments args_all_threads;
  args_all_threads.singleThread = false;
  args_all_threads.threadId = 0;

  auto result_all_threads = handler.Run(args_all_threads);
  EXPECT_THAT_EXPECTED(result_all_threads,
                       llvm::FailedWithMessage("not stopped"));

  ContinueArguments args_single_thread;
  args_single_thread.singleThread = true;
  args_single_thread.threadId = 1234;

  auto result_single_thread = handler.Run(args_single_thread);
  EXPECT_THAT_EXPECTED(result_single_thread,
                       llvm::FailedWithMessage("not stopped"));
}
