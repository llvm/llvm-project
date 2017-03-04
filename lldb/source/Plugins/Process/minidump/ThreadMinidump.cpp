//===-- ThreadMinidump.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "ThreadMinidump.h"
#include "ProcessMinidump.h"

#include "RegisterContextMinidump_x86_32.h"
#include "RegisterContextMinidump_x86_64.h"

// Other libraries and framework includes
#include "Plugins/Process/Utility/RegisterContextLinux_i386.h"
#include "Plugins/Process/Utility/RegisterContextLinux_x86_64.h"

#include "Plugins/Process/elf-core/RegisterContextPOSIXCore_x86_64.h"

#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Unwind.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"

// C Includes
// C++ Includes

using namespace lldb;
using namespace lldb_private;
using namespace minidump;

ThreadMinidump::ThreadMinidump(Process &process, const MinidumpThread &td,
                               llvm::ArrayRef<uint8_t> gpregset_data)
    : Thread(process, td.thread_id), m_thread_reg_ctx_sp(),
      m_gpregset_data(gpregset_data) {}

ThreadMinidump::~ThreadMinidump() {}

void ThreadMinidump::RefreshStateAfterStop() {}

void ThreadMinidump::ClearStackFrames() {}

RegisterContextSP ThreadMinidump::GetRegisterContext() {
  if (!m_reg_context_sp) {
    m_reg_context_sp = CreateRegisterContextForFrame(nullptr);
  }
  return m_reg_context_sp;
}

RegisterContextSP
ThreadMinidump::CreateRegisterContextForFrame(StackFrame *frame) {
  RegisterContextSP reg_ctx_sp;
  uint32_t concrete_frame_idx = 0;
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_THREAD));

  if (frame)
    concrete_frame_idx = frame->GetConcreteFrameIndex();

  if (concrete_frame_idx == 0) {
    if (m_thread_reg_ctx_sp)
      return m_thread_reg_ctx_sp;

    ProcessMinidump *process =
        static_cast<ProcessMinidump *>(GetProcess().get());
    ArchSpec arch = process->GetArchitecture();
    RegisterInfoInterface *reg_interface = nullptr;

    // TODO write other register contexts and add them here
    switch (arch.GetMachine()) {
    case llvm::Triple::x86: {
      reg_interface = new RegisterContextLinux_i386(arch);
      lldb::DataBufferSP buf =
          ConvertMinidumpContext_x86_32(m_gpregset_data, reg_interface);
      DataExtractor gpregs(buf, lldb::eByteOrderLittle, 4);
      DataExtractor fpregs;
      m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_x86_64(
          *this, reg_interface, gpregs, fpregs));
      break;
    }
    case llvm::Triple::x86_64: {
      reg_interface = new RegisterContextLinux_x86_64(arch);
      lldb::DataBufferSP buf =
          ConvertMinidumpContext_x86_64(m_gpregset_data, reg_interface);
      DataExtractor gpregs(buf, lldb::eByteOrderLittle, 8);
      DataExtractor fpregs;
      m_thread_reg_ctx_sp.reset(new RegisterContextCorePOSIX_x86_64(
          *this, reg_interface, gpregs, fpregs));
      break;
    }
    default:
      break;
    }

    if (!reg_interface) {
      if (log)
        log->Printf("elf-core::%s:: Architecture(%d) not supported",
                    __FUNCTION__, arch.GetMachine());
      assert(false && "Architecture not supported");
    }

    reg_ctx_sp = m_thread_reg_ctx_sp;
  } else if (m_unwinder_ap) {
    reg_ctx_sp = m_unwinder_ap->CreateRegisterContextForFrame(frame);
  }

  return reg_ctx_sp;
}

bool ThreadMinidump::CalculateStopInfo() { return false; }
