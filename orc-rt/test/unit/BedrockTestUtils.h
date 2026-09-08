//===- BedrockTestUtils.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test helpers that need Bedrock. These live apart from CommonTestUtils.h so
// that SupportTests translation units pull in no Bedrock headers at all.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_BEDROCKTESTUTILS_H
#define ORC_RT_UNITTEST_BEDROCKTESTUTILS_H

#include "orc-rt/bedrock/ExecutorProcessInfo.h"
#include "orc-rt/bedrock/Session.h"

#include "gtest/gtest.h"

inline orc_rt::ExecutorProcessInfo mockExecutorProcessInfo() noexcept {
  return orc_rt::ExecutorProcessInfo("arm64-apple-darwin", 16384,
                                     "+neon, +fullfp16");
}

/// DispatchFn for tests that should never dispatch a task. Records a test
/// failure on invocation, then runs the task inline so that any caller
/// awaiting a result unblocks (rather than hanging) and the keepalive token
/// is released, even in -Asserts builds or when the dispatch arrives on a
/// non-test thread.
inline void noDispatch(orc_rt::Session::Task T) {
  ADD_FAILURE() << "unexpected dispatch in a no-dispatch session";
  T();
}

/// DispatchFn that runs tasks on the current thread.
inline void inlineDispatch(orc_rt::Session::Task T) { T(); }

#endif // ORC_RT_UNITTEST_BEDROCKTESTUTILS_H
