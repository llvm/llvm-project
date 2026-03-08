//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_EVALUATE_CONTEXT_H
#define LLDB_TOOLS_LLDB_DAP_EVALUATE_CONTEXT_H

#include "DAPForward.h"
#include "lldb/API/SBValueList.h"
#include "lldb/Host/MainLoop.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace lldb_dap {
class EvaluateContext {
public:
  bool HandleOutput(llvm::StringRef);
  bool HandleReturnObject(lldb::SBCommandReturnObject &);
  static llvm::Expected<std::pair<std::string, lldb::SBValueList>>
  Run(DAP &dap, llvm::StringRef expr);

  void Interrupt();

private:
  bool WantsRawInput();
  void Done(bool immediate);
  EvaluateContext(DAP &dap, llvm::StringRef expr);

  DAP &m_dap;
  llvm::StringRef m_expr;
  llvm::SmallString<32> m_output;
  lldb::SBValueList m_variables;
  bool m_echo_detected = false;
  bool m_success = true;
  bool m_return_object_reported = false;
  // If the CommandInterpreter is not active, then we're in raw input mode and
  // will not receive a result from the print object helper.
  bool m_wants_return_object = false;
  lldb_private::MainLoop m_loop;
};

} // namespace lldb_dap

#endif
