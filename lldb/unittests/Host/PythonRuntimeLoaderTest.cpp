//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "lldb/Host/PythonRuntimeLoader.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

// PythonRuntimeLoader::Load is a process-global once_flag, so we can exercise
// it only once per process. Verify that subsequent calls return a result
// consistent with the first.
TEST(PythonRuntimeLoaderTest, LoadIsIdempotent) {
  llvm::Error first = PythonRuntimeLoader::Load();
  bool first_failed = static_cast<bool>(first);
  llvm::consumeError(std::move(first));

  llvm::Error second = PythonRuntimeLoader::Load();
  bool second_failed = static_cast<bool>(second);
  llvm::consumeError(std::move(second));

  EXPECT_EQ(first_failed, second_failed);
}

#endif // LLDB_ENABLE_PYTHON
