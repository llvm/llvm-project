//===-- RegisterContextMinidump_x86_64.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextMinidump_x86_64.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"

#include "lldb/Utility/DataBufferHeap.h"

// C includes
// C++ includes


namespace reg_ids = lldb_private::registers::x86_64::id;
using namespace lldb_private;
using namespace minidump;

static llvm::MutableArrayRef<uint8_t> getDestRegister(uint8_t *context,
                                                      const RegisterInfo &reg) {
  auto bytes = reg.mutable_data(context);

  switch (reg.kinds[lldb::eRegisterKindLLDB]) {
  case reg_ids::cs:
  case reg_ids::ds:
  case reg_ids::es:
  case reg_ids::fs:
  case reg_ids::gs:
  case reg_ids::ss:
    return bytes.take_front(2);
    break;
  case reg_ids::rflags:
    return bytes.take_front(4);
    break;
  default:
    return bytes.take_front(8);
    break;
  }
}

static void writeRegister(const void *reg_src, uint8_t *context,
                          const RegisterInfo &reg) {
  llvm::MutableArrayRef<uint8_t> reg_dest = getDestRegister(context, reg);
  memcpy(reg_dest.data(), reg_src, reg_dest.size());
}

// TODO: Fix the registers in this file!
// writeRegister checks x86_64 registers without base registers. This causes
// an overlap in the register enum values. So we were truncating fs_base.
// We should standardize to the x86_64_with_base registers.
static void writeBaseRegister(const void *reg_src, uint8_t *context,
                              const RegisterInfo &reg) {
  auto bytes = reg.mutable_data(context);
  llvm::MutableArrayRef<uint8_t> reg_dest = bytes.take_front(8);
  memcpy(reg_dest.data(), reg_src, reg_dest.size());
}

lldb::DataBufferSP lldb_private::minidump::ConvertMinidumpContext_x86_64(
    llvm::ArrayRef<uint8_t> source_data,
    RegisterInfoInterface *target_reg_interface) {

  const RegisterInfo *reg_info = target_reg_interface->GetRegisterInfo();

  lldb::WritableDataBufferSP result_context_buf(
      new DataBufferHeap(target_reg_interface->GetGPRSize(), 0));
  uint8_t *result_base = result_context_buf->GetBytes();

  if (source_data.size() < sizeof(MinidumpContext_x86_64))
    return nullptr;

  const MinidumpContext_x86_64 *context;
  consumeObject(source_data, context);

  const MinidumpContext_x86_64_Flags context_flags =
      static_cast<MinidumpContext_x86_64_Flags>(
          static_cast<uint32_t>(context->context_flags));
  auto x86_64_Flag = MinidumpContext_x86_64_Flags::x86_64_Flag;
  auto ControlFlag = MinidumpContext_x86_64_Flags::Control;
  auto IntegerFlag = MinidumpContext_x86_64_Flags::Integer;
  auto SegmentsFlag = MinidumpContext_x86_64_Flags::Segments;
  auto LLDBSpecificFlag = MinidumpContext_x86_64_Flags::LLDBSpecific;

  if ((context_flags & x86_64_Flag) != x86_64_Flag)
    return nullptr;

  if ((context_flags & ControlFlag) == ControlFlag) {
    writeRegister(&context->cs, result_base, reg_info[reg_ids::cs]);
    writeRegister(&context->ss, result_base, reg_info[reg_ids::ss]);
    writeRegister(&context->eflags, result_base, reg_info[reg_ids::rflags]);
    writeRegister(&context->rsp, result_base, reg_info[reg_ids::rsp]);
    writeRegister(&context->rip, result_base, reg_info[reg_ids::rip]);
  }

  if ((context_flags & SegmentsFlag) == SegmentsFlag) {
    writeRegister(&context->ds, result_base, reg_info[reg_ids::ds]);
    writeRegister(&context->es, result_base, reg_info[reg_ids::es]);
    writeRegister(&context->fs, result_base, reg_info[reg_ids::fs]);
    writeRegister(&context->gs, result_base, reg_info[reg_ids::gs]);
  }

  if ((context_flags & IntegerFlag) == IntegerFlag) {
    writeRegister(&context->rax, result_base, reg_info[reg_ids::rax]);
    writeRegister(&context->rcx, result_base, reg_info[reg_ids::rcx]);
    writeRegister(&context->rdx, result_base, reg_info[reg_ids::rdx]);
    writeRegister(&context->rbx, result_base, reg_info[reg_ids::rbx]);
    writeRegister(&context->rbp, result_base, reg_info[reg_ids::rbp]);
    writeRegister(&context->rsi, result_base, reg_info[reg_ids::rsi]);
    writeRegister(&context->rdi, result_base, reg_info[reg_ids::rdi]);
    writeRegister(&context->r8, result_base, reg_info[reg_ids::r8]);
    writeRegister(&context->r9, result_base, reg_info[reg_ids::r9]);
    writeRegister(&context->r10, result_base, reg_info[reg_ids::r10]);
    writeRegister(&context->r11, result_base, reg_info[reg_ids::r11]);
    writeRegister(&context->r12, result_base, reg_info[reg_ids::r12]);
    writeRegister(&context->r13, result_base, reg_info[reg_ids::r13]);
    writeRegister(&context->r14, result_base, reg_info[reg_ids::r14]);
    writeRegister(&context->r15, result_base, reg_info[reg_ids::r15]);
  }

  // See comment on base register
  if ((context_flags & LLDBSpecificFlag) == LLDBSpecificFlag) {
    writeBaseRegister(&context->fs_base, result_base,
                      reg_info[reg_ids::with_base::fs_base]);
    writeBaseRegister(&context->gs_base, result_base,
                      reg_info[reg_ids::with_base::gs_base]);
  }

  // TODO parse the floating point registers

  return result_context_buf;
}
