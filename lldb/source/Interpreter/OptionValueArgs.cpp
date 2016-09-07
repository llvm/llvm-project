//===-- OptionValueArgs.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValueArgs.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

size_t OptionValueArgs::GetArgs(Args &args) {
  const uint32_t size = m_values.size();
  std::vector<const char *> argv;
  for (uint32_t i = 0; i < size; ++i) {
    const char *string_value = m_values[i]->GetStringValue();
    if (string_value)
      argv.push_back(string_value);
  }

  if (argv.empty())
    args.Clear();
  else
    args.SetArguments(argv.size(), &argv[0]);
  return args.GetArgumentCount();
}
