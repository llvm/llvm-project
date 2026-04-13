//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadWasm.h"

#include "ProcessWasm.h"
#include "RegisterContextWasm.h"
#include "UnwindWasm.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::wasm;

Unwind &ThreadWasm::GetUnwinder() {
  if (!m_unwinder_up) {
    assert(CalculateTarget()->GetArchitecture().GetMachine() ==
           llvm::Triple::wasm32);
    m_unwinder_up.reset(new wasm::UnwindWasm(*this));
  }
  return *m_unwinder_up;
}

llvm::Expected<std::vector<lldb::addr_t>> ThreadWasm::GetWasmCallStack() {
  if (ProcessSP process_sp = GetProcess()) {
    ProcessWasm *wasm_process = static_cast<ProcessWasm *>(process_sp.get());
    return wasm_process->GetWasmCallStack(GetID());
  }
  return llvm::createStringError("no process");
}

lldb::RegisterContextSP
ThreadWasm::CreateRegisterContextForFrame(StackFrame *frame) {
  uint32_t concrete_frame_idx = 0;
  ProcessSP process_sp(GetProcess());
  ProcessWasm *wasm_process = static_cast<ProcessWasm *>(process_sp.get());

  if (frame)
    concrete_frame_idx = frame->GetConcreteFrameIndex();

  if (concrete_frame_idx == 0)
    return std::make_shared<RegisterContextWasm>(
        *this, concrete_frame_idx, wasm_process->GetRegisterInfo());

  return GetUnwinder().CreateRegisterContextForFrame(frame);
}
