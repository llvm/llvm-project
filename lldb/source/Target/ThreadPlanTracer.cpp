//===-- ThreadPlanTracer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/DumpRegisterValue.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark ThreadPlanTracer

ThreadPlanTracer::ThreadPlanTracer(Thread &thread, lldb::StreamSP &stream_sp)
    : m_thread(thread), m_single_step(true), m_enabled(false),
      m_stream_sp(stream_sp) {}

ThreadPlanTracer::ThreadPlanTracer(Thread &thread)
    : m_thread(thread), m_single_step(true), m_enabled(false), m_stream_sp() {}

Stream *ThreadPlanTracer::GetLogStream() {
  if (m_stream_sp)
    return m_stream_sp.get();
  else {
    TargetSP target_sp(m_thread.CalculateTarget());
    if (target_sp)
      return &(target_sp->GetDebugger().GetOutputStream());
  }
  return nullptr;
}

void ThreadPlanTracer::Log() {
  SymbolContext sc;
  bool show_frame_index = false;
  bool show_fullpaths = false;

  Stream *stream = GetLogStream();
  if (stream) {
    m_thread.GetStackFrameAtIndex(0)->Dump(stream, show_frame_index,
                                           show_fullpaths);
    stream->Printf("\n");
    stream->Flush();
  }
}

bool ThreadPlanTracer::TracerExplainsStop() {
  if (m_enabled && m_single_step) {
    lldb::StopInfoSP stop_info = m_thread.GetStopInfo();
    return (stop_info->GetStopReason() == eStopReasonTrace);
  } else
    return false;
}

#pragma mark ThreadPlanAssemblyTracer

ThreadPlanAssemblyTracer::ThreadPlanAssemblyTracer(Thread &thread,
                                                   lldb::StreamSP &stream_sp)
    : ThreadPlanTracer(thread, stream_sp), m_disassembler_sp(), m_intptr_type(),
      m_register_values() {}

ThreadPlanAssemblyTracer::ThreadPlanAssemblyTracer(Thread &thread)
    : ThreadPlanTracer(thread), m_disassembler_sp(), m_intptr_type(),
      m_register_values() {}

Disassembler *ThreadPlanAssemblyTracer::GetDisassembler() {
  if (!m_disassembler_sp)
    m_disassembler_sp = Disassembler::FindPlugin(
        m_thread.GetProcess()->GetTarget().GetArchitecture(), nullptr, nullptr);
  return m_disassembler_sp.get();
}

TypeFromUser ThreadPlanAssemblyTracer::GetIntPointerType() {
  if (!m_intptr_type.IsValid()) {
    if (auto target_sp = m_thread.CalculateTarget()) {
      auto type_system_or_err =
          target_sp->GetScratchTypeSystemForLanguage(eLanguageTypeC);
      if (auto err = type_system_or_err.takeError()) {
        LLDB_LOG_ERROR(
            lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_TYPES),
            std::move(err),
            "Unable to get integer pointer type from TypeSystem");
      } else {
        m_intptr_type = TypeFromUser(
            type_system_or_err->GetBuiltinTypeForEncodingAndBitSize(
                eEncodingUint,
                target_sp->GetArchitecture().GetAddressByteSize() * 8));
      }
    }
  }
  return m_intptr_type;
}

ThreadPlanAssemblyTracer::~ThreadPlanAssemblyTracer() = default;

void ThreadPlanAssemblyTracer::TracingStarted() {
}

void ThreadPlanAssemblyTracer::TracingEnded() { m_register_values.clear(); }

void ThreadPlanAssemblyTracer::Log() {
  Stream *stream = GetLogStream();

  if (!stream)
    return;

  RegisterContext *reg_ctx = m_thread.GetRegisterContext().get();

  lldb::addr_t pc = reg_ctx->GetPC();
  ProcessSP process_sp(m_thread.GetProcess());
  Address pc_addr;
  bool addr_valid = false;
  uint8_t buffer[16] = {0}; // Must be big enough for any single instruction
  addr_valid = process_sp->GetTarget().GetSectionLoadList().ResolveLoadAddress(
      pc, pc_addr);

  pc_addr.Dump(stream, &m_thread, Address::DumpStyleResolvedDescription,
               Address::DumpStyleModuleWithFileAddress);
  stream->PutCString(" ");

  Disassembler *disassembler = GetDisassembler();
  if (disassembler) {
    Status err;
    process_sp->ReadMemory(pc, buffer, sizeof(buffer), err);

    if (err.Success()) {
      DataExtractor extractor(buffer, sizeof(buffer),
                              process_sp->GetByteOrder(),
                              process_sp->GetAddressByteSize());

      bool data_from_file = false;
      if (addr_valid)
        disassembler->DecodeInstructions(pc_addr, extractor, 0, 1, false,
                                         data_from_file);
      else
        disassembler->DecodeInstructions(Address(pc), extractor, 0, 1, false,
                                         data_from_file);

      InstructionList &instruction_list = disassembler->GetInstructionList();
      const uint32_t max_opcode_byte_size =
          instruction_list.GetMaxOpcocdeByteSize();

      if (instruction_list.GetSize()) {
        const bool show_bytes = true;
        const bool show_address = true;
        Instruction *instruction =
            instruction_list.GetInstructionAtIndex(0).get();
        const FormatEntity::Entry *disassemble_format =
            m_thread.GetProcess()
                ->GetTarget()
                .GetDebugger()
                .GetDisassemblyFormat();
        instruction->Dump(stream, max_opcode_byte_size, show_address,
                          show_bytes, nullptr, nullptr, nullptr,
                          disassemble_format, 0);
      }
    }
  }

  const ABI *abi = process_sp->GetABI().get();
  TypeFromUser intptr_type = GetIntPointerType();

  if (abi && intptr_type.IsValid()) {
    ValueList value_list;
    const int num_args = 1;

    for (int arg_index = 0; arg_index < num_args; ++arg_index) {
      Value value;
      value.SetValueType(Value::eValueTypeScalar);
      value.SetCompilerType(intptr_type);
      value_list.PushValue(value);
    }

    if (abi->GetArgumentValues(m_thread, value_list)) {
      for (int arg_index = 0; arg_index < num_args; ++arg_index) {
        stream->Printf(
            "\n\targ[%d]=%llx", arg_index,
            value_list.GetValueAtIndex(arg_index)->GetScalar().ULongLong());

        if (arg_index + 1 < num_args)
          stream->PutCString(", ");
      }
    }
  }

  if (m_register_values.empty()) {
    RegisterContext *reg_ctx = m_thread.GetRegisterContext().get();
    m_register_values.resize(reg_ctx->GetRegisterCount());
  }

  RegisterValue reg_value;
  for (uint32_t reg_num = 0, num_registers = reg_ctx->GetRegisterCount();
       reg_num < num_registers; ++reg_num) {
    const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoAtIndex(reg_num);
    if (reg_ctx->ReadRegister(reg_info, reg_value)) {
      assert(reg_num < m_register_values.size());
      if (m_register_values[reg_num].GetType() == RegisterValue::eTypeInvalid ||
          reg_value != m_register_values[reg_num]) {
        if (reg_value.GetType() != RegisterValue::eTypeInvalid) {
          stream->PutCString("\n\t");
          DumpRegisterValue(reg_value, stream, reg_info, true, false,
                            eFormatDefault);
        }
      }
      m_register_values[reg_num] = reg_value;
    }
  }
  stream->EOL();
  stream->Flush();
}
