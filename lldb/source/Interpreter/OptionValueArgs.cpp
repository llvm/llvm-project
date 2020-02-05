//===-- OptionValueArgs.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueArgs.h"

#include "lldb/Utility/Args.h"

using namespace lldb;
using namespace lldb_private;

size_t OptionValueArgs::GetArgs(Args &args) {
  args.Clear();
  for (auto value : m_values) {
    llvm::StringRef string_value = value->GetStringValue();
    if (!string_value.empty())
      args.AppendArgument(string_value);
  }

  return args.GetArgumentCount();
}
