//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EvaluateContext.h"
#include "DAP.h"
#include "DAPLog.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBMutex.h"
#include "lldb/API/SBProgress.h"
#include "lldb/Host/File.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <chrono>
#include <memory>
#include <string>

using namespace llvm;
using namespace lldb_dap;
using lldb_private::Status;

EvaluateContext::EvaluateContext(DAP &dap, StringRef expr)
    : m_dap(dap), m_expr(expr),
      m_wants_return_object(dap.debugger.GetCommandInterpreter().IsActive()) {}

void EvaluateContext::Interrupt() {
  DAP_LOG(m_dap.log, "EvaluateContext::Interrupt");
  Done(/*immediate=*/true);
}

bool EvaluateContext::WantsRawInput() {
  return !m_dap.debugger.GetCommandInterpreter().IsActive();
}

void EvaluateContext::Done(bool immediate) {
  DAP_LOG(m_dap.log, "EvaluateContext::Done");
  if (immediate)
    m_loop.AddPendingCallback([](auto &loop) { loop.RequestTermination(); });
  else
    m_loop.AddCallback([](auto &loop) { loop.RequestTermination(); },
                       std::chrono::milliseconds(50));
}

bool EvaluateContext::HandleOutput(StringRef o) {
  DAP_LOG(m_dap.log, "EvaluateContext::HandleOutput(o={0})", o);
  // Skip the echo of the input
  if (o.trim() == m_expr.trim() && !m_echo_detected)
    m_echo_detected = true;
  else
    m_output += o;

  if (m_wants_return_object) {
    if (m_return_object_reported)
      Done(/*immediate=*/m_output.ends_with(m_dap.debugger.GetPrompt()));
  } else {
    Done(/*immediate=*/m_output.ends_with(m_dap.debugger.GetPrompt()));
  }

  return true;
}

bool EvaluateContext::HandleReturnObject(lldb::SBCommandReturnObject &result) {
  DAP_LOG(m_dap.log, "EvaluateContext::HandleReturnObject");
  m_success = result.Succeeded();

  m_return_object_reported = true;

  if (result.GetStatus() == lldb::eReturnStatusSuccessFinishResult)
    m_variables = result.GetValues(lldb::eDynamicDontRunTarget);
  if (result.GetOutputSize())
    m_output += StringRef{result.GetOutput(), result.GetOutputSize()};
  if (result.GetErrorSize())
    m_output += StringRef{result.GetError(), result.GetErrorSize()};

  if (WantsRawInput())
    Done(/*immediate=*/false);
  else
    Done(/*immediate=*/!m_output.empty());

  return true;
}

Expected<std::pair<std::string, lldb::SBValueList>>
EvaluateContext::Run(DAP &dap, StringRef expr) {
  DAP_LOG(dap.log, "EvaluateContext::Run");
  EvaluateContext context(dap, expr);
  dap.SetEvaluateContext(&context);

  lldb::SBProgress progress(/*title=*/"Evaluating expression",
                            expr.str().data(), dap.debugger);

  lldb::SBMutex api_mutex = dap.GetAPIMutex();

  // While in raw input mode, don't wait for output in case the IOHandler never
  // writes.
  if (context.WantsRawInput() && !dap.primary.write->GetIsInteractive())
    context.Done(false);

  // Unlock to allow the background thread to handle reading/processing.
  api_mutex.unlock();

  size_t size = expr.size();
  if (Error err = dap.primary.write->Write(expr.data(), size).takeError())
    return err;

  Status status = context.m_loop.Run();

  api_mutex.lock();
  dap.SetEvaluateContext(nullptr);

  if (auto error = status.takeError())
    return error;

  if (!context.m_success)
    return createStringError(context.m_output.empty()
                                 ? "Error evaluating expression"
                                 : context.m_output.str());

  return std::make_pair(context.m_output.str().str(), context.m_variables);
}
