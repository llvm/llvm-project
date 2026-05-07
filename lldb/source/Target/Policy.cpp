//===-- Policy.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Policy.h"
#include "lldb/Utility/Stream.h"

using namespace lldb_private;

void Policy::Dump(Stream &s) const {
  s.Printf("view=%s", view == View::Public ? "public" : "private");
  s.PutCString(", capabilities={");
  s.Printf("eval_expr=%d", capabilities.can_evaluate_expressions);
  s.Printf(" run_all=%d", capabilities.can_run_all_threads);
  s.Printf(" try_all=%d", capabilities.can_try_all_threads);
  s.Printf(" bp_actions=%d", capabilities.can_run_breakpoint_actions);
  s.Printf(" frame_providers=%d", capabilities.can_load_frame_providers);
  s.Printf(" frame_recognizers=%d", capabilities.can_run_frame_recognizers);
  s.PutChar('}');
}

void PolicyStack::Dump(Stream &s) const {
  s.Printf("PolicyStack depth=%zu\n", m_stack.size());
  for (size_t i = 0; i < m_stack.size(); i++) {
    s.Printf("  [%zu] ", i);
    m_stack[i].Dump(s);
    s.PutChar('\n');
  }
}
