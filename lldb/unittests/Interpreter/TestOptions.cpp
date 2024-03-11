//===-- TestOptions.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/Options.h"
#include "gtest/gtest.h"

#include "llvm/Testing/Support/Error.h"

using namespace lldb_private;

TEST(OptionsTest, CreateOptionParsingError) {
  ASSERT_THAT_ERROR(
      CreateOptionParsingError("yippee", 'f', "fun",
                               "unable to convert 'yippee' to boolean"),
      llvm::FailedWithMessage("Invalid value ('yippee') for -f (fun): unable "
                              "to convert 'yippee' to boolean"));

  ASSERT_THAT_ERROR(
      CreateOptionParsingError("52", 'b', "bean-count"),
      llvm::FailedWithMessage("Invalid value ('52') for -b (bean-count)"));

  ASSERT_THAT_ERROR(CreateOptionParsingError("c", 'm'),
                    llvm::FailedWithMessage("Invalid value ('c') for -m"));
}
