//===-- OptionGroupArchitecture.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionGroupArchitecture.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"
#include "lldb/Target/Platform.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_architecture
#include "InterpreterOptions.inc"

llvm::ArrayRef<OptionDefinition> OptionGroupArchitecture::GetDefinitions() {
  return llvm::ArrayRef(g_architecture_options);
}

bool OptionGroupArchitecture::GetArchitecture(Platform *platform,
                                              ArchSpec &arch) {
  arch = Platform::GetAugmentedArchSpec(platform, m_arch_str);
  return arch.IsValid();
}

Status
OptionGroupArchitecture::SetOptionValue(uint32_t option_idx,
                                        llvm::StringRef option_arg,
                                        ExecutionContext *execution_context) {
  Status error;
  const int short_option = g_architecture_options[option_idx].short_option;

  switch (short_option) {
  case 'a':
    m_arch_str.assign(std::string(option_arg));
    break;

  default:
    llvm_unreachable("Unimplemented option");
  }

  return error;
}

void OptionGroupArchitecture::OptionParsingStarting(
    ExecutionContext *execution_context) {
  m_arch_str.clear();
}
