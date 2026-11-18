//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadFreeBSDKernelCore.h"

#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Unwind.h"
#include "lldb/Utility/Log.h"

#include "ProcessFreeBSDKernelCore.h"
#include "RegisterContextFreeBSDKernelCore_arm.h"
#include "RegisterContextFreeBSDKernelCore_arm64.h"
#include "RegisterContextFreeBSDKernelCore_i386.h"
#include "RegisterContextFreeBSDKernelCore_ppc64le.h"
#include "RegisterContextFreeBSDKernelCore_riscv64.h"
#include "RegisterContextFreeBSDKernelCore_x86_64.h"
#include "lldb/Register/Descriptions/RegisterDescription_arm32.h"
#include "lldb/Register/Descriptions/RegisterDescription_arm64.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc64le.h"
#include "lldb/Register/Descriptions/RegisterDescription_riscv64.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_x86_32.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_x86_64.h"

using namespace lldb;
using namespace lldb_private;

ThreadFreeBSDKernelCore::ThreadFreeBSDKernelCore(Process &process,
                                                 lldb::tid_t tid,
                                                 lldb::addr_t pcb_addr,
                                                 std::string thread_name)
    : Thread(process, tid), m_thread_name(std::move(thread_name)),
      m_pcb_addr(pcb_addr) {}

ThreadFreeBSDKernelCore::~ThreadFreeBSDKernelCore() {}

void ThreadFreeBSDKernelCore::RefreshStateAfterStop() {}

lldb::ThreadRegisterContextSP ThreadFreeBSDKernelCore::GetRegisterContext() {
  if (!m_reg_context_sp)
    m_reg_context_sp = CreateRegisterContextForFrame(nullptr);
  return m_reg_context_sp;
}

lldb::ThreadRegisterContextSP
ThreadFreeBSDKernelCore::CreateRegisterContextForFrame(StackFrame *frame) {
  ThreadRegisterContextSP reg_ctx_sp;
  uint32_t concrete_frame_idx = 0;

  if (frame)
    concrete_frame_idx = frame->GetConcreteFrameIndex();

  if (concrete_frame_idx == 0) {
    if (m_thread_reg_ctx_sp)
      return m_thread_reg_ctx_sp;

    ProcessFreeBSDKernelCore *process =
        static_cast<ProcessFreeBSDKernelCore *>(GetProcess().get());
    ArchSpec arch = process->GetTarget().GetArchitecture();

    switch (arch.GetMachine()) {
    case llvm::Triple::aarch64: {
      RegisterLayout layout;
      llvm::cantFail(
          layout.Finalize(std::make_shared<RegisterDescription_arm64>(
                              RegisterDescription_arm64::Config{0}),
                          arch.GetByteOrder()));
      m_thread_reg_ctx_sp =
          std::make_shared<RegisterContextFreeBSDKernelCore_arm64>(
              *this, std::move(layout), m_pcb_addr);
      break;
    }
    case llvm::Triple::arm: {
      RegisterLayout layout;
      llvm::cantFail(layout.Finalize(RegisterDescription_arm32::GetShared(),
                                     arch.GetByteOrder()));
      m_thread_reg_ctx_sp =
          std::make_shared<RegisterContextFreeBSDKernelCore_arm>(
              *this, std::move(layout), m_pcb_addr);
      break;
    }
    case llvm::Triple::ppc64le: {
      RegisterLayout layout;
      llvm::cantFail(layout.Finalize(RegisterDescription_ppc64le::GetShared(),
                                     arch.GetByteOrder()));
      m_thread_reg_ctx_sp =
          std::make_shared<RegisterContextFreeBSDKernelCore_ppc64le>(
              *this, std::move(layout), m_pcb_addr);
      break;
    }
    case llvm::Triple::riscv64: {
      RegisterLayout layout;
      llvm::cantFail(
          layout.Finalize(RegisterDescription_riscv64::GetShared(
                              RegisterDescription_riscv64::Config{0}),
                          arch.GetByteOrder()));
      m_thread_reg_ctx_sp =
          std::make_shared<RegisterContextFreeBSDKernelCore_riscv64>(
              *this, std::move(layout), m_pcb_addr);
      break;
    }
    case llvm::Triple::x86:
      m_thread_reg_ctx_sp =
          std::make_shared<RegisterContextFreeBSDKernelCore_i386>(
              *this, RegisterLayoutFreeBSD_x86_32(arch), m_pcb_addr);
      break;
    case llvm::Triple::x86_64:
      m_thread_reg_ctx_sp =
          std::make_shared<RegisterContextFreeBSDKernelCore_x86_64>(
              *this, RegisterLayoutFreeBSD_x86_64(arch), m_pcb_addr);
      break;
    default:
      assert(false &&
             "Unsupported architecture passed to ThreadFreeBSDKernelCore");
      break;
    }

    reg_ctx_sp = m_thread_reg_ctx_sp;
  } else {
    reg_ctx_sp = GetUnwinder().CreateRegisterContextForFrame(frame);
  }
  return reg_ctx_sp;
}

bool ThreadFreeBSDKernelCore::CalculateStopInfo() {
  if (m_is_crashed) {
    // Set a stop reason for crashing threads only so that they get selected
    // preferentially.
    SetStopInfo(StopInfo::CreateStopReasonWithException(*this, "kernel panic"));
    return true;
  }
  return false;
}
