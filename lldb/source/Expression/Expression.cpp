//===-- Expression.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/Expression.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

using namespace lldb_private;

Expression::Expression(Target &target)
    : m_target_wp(target.shared_from_this()),
      m_jit_start_addr(LLDB_INVALID_ADDRESS),
      m_jit_end_addr(LLDB_INVALID_ADDRESS) {
  // Can't make any kind of expression without a target.
  assert(m_target_wp.lock());
}

Expression::Expression(ExecutionContextScope &exe_scope)
    : m_target_wp(exe_scope.CalculateTarget()),
      m_jit_start_addr(LLDB_INVALID_ADDRESS),
      m_jit_end_addr(LLDB_INVALID_ADDRESS) {
  assert(m_target_wp.lock());
}

bool lldb_private::consumeFunctionCallLabelPrefix(llvm::StringRef &name) {
  // On Darwin mangled names get a '_' prefix.
  name.consume_front("_");
  return name.consume_front(FunctionCallLabelPrefix);
}

bool lldb_private::hasFunctionCallLabelPrefix(llvm::StringRef name) {
  // On Darwin mangled names get a '_' prefix.
  name.consume_front("_");
  return name.starts_with(FunctionCallLabelPrefix);
}
