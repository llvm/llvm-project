//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "lldb/Host/ScriptInterpreterRuntimeLoader.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

// Load() uses a process-wide once_flag, so this is the only test that can
// exercise it in this process.
TEST(PythonRuntimeLoaderTest, LoadIsIdempotent) {
  llvm::Expected<ScriptInterpreterRuntimeLoader &> loader =
      ScriptInterpreterRuntimeLoader::Get(lldb::eScriptLanguagePython);
  ASSERT_TRUE(static_cast<bool>(loader));

  llvm::Error first = loader->Load();
  const bool first_result = static_cast<bool>(first);
  llvm::consumeError(std::move(first));

  llvm::Error second = loader->Load();
  const bool second_result = static_cast<bool>(second);
  llvm::consumeError(std::move(second));

  EXPECT_EQ(first_result, second_result);
}

#endif // LLDB_ENABLE_PYTHON
