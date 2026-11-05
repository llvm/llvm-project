//===-- RegisterContextMinidump_x86_32.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextMinidump_x86_32.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"

#include "lldb/Utility/DataBufferHeap.h"

// C includes
// C++ includes


namespace reg_ids = lldb_private::registers::x86::id;
using namespace lldb_private;
using namespace minidump;

static void writeRegister(const void *reg_src,
                          llvm::MutableArrayRef<uint8_t> reg_dest) {
  memcpy(reg_dest.data(), reg_src, reg_dest.size());
}

lldb::DataBufferSP lldb_private::minidump::ConvertMinidumpContext_x86_32(
    llvm::ArrayRef<uint8_t> source_data,
    RegisterInfoInterface *target_reg_interface) {

  const RegisterInfo *reg_info = target_reg_interface->GetRegisterInfo();

  lldb::WritableDataBufferSP result_context_buf(
      new DataBufferHeap(target_reg_interface->GetGPRSize(), 0));
  uint8_t *result_base = result_context_buf->GetBytes();

  if (source_data.size() < sizeof(MinidumpContext_x86_32))
    return nullptr;

  const MinidumpContext_x86_32 *context;
  consumeObject(source_data, context);

  const MinidumpContext_x86_32_Flags context_flags =
      static_cast<MinidumpContext_x86_32_Flags>(
          static_cast<uint32_t>(context->context_flags));
  auto x86_32_Flag = MinidumpContext_x86_32_Flags::x86_32_Flag;
  auto ControlFlag = MinidumpContext_x86_32_Flags::Control;
  auto IntegerFlag = MinidumpContext_x86_32_Flags::Integer;
  auto SegmentsFlag = MinidumpContext_x86_32_Flags::Segments;

  if ((context_flags & x86_32_Flag) != x86_32_Flag) {
    return nullptr;
  }

  if ((context_flags & ControlFlag) == ControlFlag) {
    writeRegister(&context->ebp,
                  reg_info[reg_ids::ebp].mutable_data(result_base));
    writeRegister(&context->eip,
                  reg_info[reg_ids::eip].mutable_data(result_base));
    writeRegister(&context->cs,
                  reg_info[reg_ids::cs].mutable_data(result_base));
    writeRegister(&context->eflags,
                  reg_info[reg_ids::eflags].mutable_data(result_base));
    writeRegister(&context->esp,
                  reg_info[reg_ids::esp].mutable_data(result_base));
    writeRegister(&context->ss,
                  reg_info[reg_ids::ss].mutable_data(result_base));
  }

  if ((context_flags & SegmentsFlag) == SegmentsFlag) {
    writeRegister(&context->ds,
                  reg_info[reg_ids::ds].mutable_data(result_base));
    writeRegister(&context->es,
                  reg_info[reg_ids::es].mutable_data(result_base));
    writeRegister(&context->fs,
                  reg_info[reg_ids::fs].mutable_data(result_base));
    writeRegister(&context->gs,
                  reg_info[reg_ids::gs].mutable_data(result_base));
  }

  if ((context_flags & IntegerFlag) == IntegerFlag) {
    writeRegister(&context->eax,
                  reg_info[reg_ids::eax].mutable_data(result_base));
    writeRegister(&context->ecx,
                  reg_info[reg_ids::ecx].mutable_data(result_base));
    writeRegister(&context->edx,
                  reg_info[reg_ids::edx].mutable_data(result_base));
    writeRegister(&context->ebx,
                  reg_info[reg_ids::ebx].mutable_data(result_base));
    writeRegister(&context->esi,
                  reg_info[reg_ids::esi].mutable_data(result_base));
    writeRegister(&context->edi,
                  reg_info[reg_ids::edi].mutable_data(result_base));
  }

  // TODO parse the floating point registers

  return result_context_buf;
}
