//===-- RegisterContextMinidump_x86_64.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "RegisterContextMinidump_x86_64.h"

// Other libraries and framework includes
#include "lldb/Core/DataBufferHeap.h"

// C includes
// C++ includes

using namespace lldb_private;
using namespace minidump;

void writeRegister(llvm::ArrayRef<uint8_t> &reg_src,
                   llvm::MutableArrayRef<uint8_t> reg_dest) {
  memcpy(reg_dest.data(), reg_src.data(), reg_dest.size());
  reg_src = reg_src.drop_front(reg_dest.size());
}

llvm::MutableArrayRef<uint8_t> getDestRegister(uint8_t *context,
                                               uint32_t lldb_reg_num,
                                               const RegisterInfo &reg) {
  auto bytes = reg.mutable_data(context);

  switch (lldb_reg_num) {
  case lldb_cs_x86_64:
  case lldb_ds_x86_64:
  case lldb_es_x86_64:
  case lldb_fs_x86_64:
  case lldb_gs_x86_64:
  case lldb_ss_x86_64:
    return bytes.take_front(2);
    break;
  case lldb_rflags_x86_64:
    return bytes.take_front(4);
    break;
  default:
    return bytes.take_front(8);
    break;
  }
}

lldb::DataBufferSP lldb_private::minidump::ConvertMinidumpContextToRegIface(
    llvm::ArrayRef<uint8_t> source_data,
    RegisterInfoInterface *target_reg_interface) {

  const RegisterInfo *reg_info = target_reg_interface->GetRegisterInfo();

  lldb::DataBufferSP result_context_buf(
      new DataBufferHeap(target_reg_interface->GetGPRSize(), 0));
  uint8_t *result_base = result_context_buf->GetBytes();

  source_data = source_data.drop_front(6 * 8); // p[1-6] home registers
  const uint32_t *context_flags;
  consumeObject(source_data, context_flags);
  const uint32_t x86_64_Flag =
      static_cast<uint32_t>(MinidumpContext_x86_64_Flags::x86_64_Flag);
  const uint32_t ControlFlag =
      static_cast<uint32_t>(MinidumpContext_x86_64_Flags::Control);
  const uint32_t IntegerFlag =
      static_cast<uint32_t>(MinidumpContext_x86_64_Flags::Integer);
  const uint32_t SegmentsFlag =
      static_cast<uint32_t>(MinidumpContext_x86_64_Flags::Segments);
  const uint32_t DebugRegistersFlag =
      static_cast<uint32_t>(MinidumpContext_x86_64_Flags::DebugRegisters);

  if (!(*context_flags & x86_64_Flag)) {
    return result_context_buf; // error
  }

  source_data = source_data.drop_front(4); // mx_csr

  if (*context_flags & ControlFlag) {
    writeRegister(source_data, getDestRegister(result_base, lldb_cs_x86_64,
                                               reg_info[lldb_cs_x86_64]));
  }

  if (*context_flags & SegmentsFlag) {
    writeRegister(source_data, getDestRegister(result_base, lldb_ds_x86_64,
                                               reg_info[lldb_ds_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_es_x86_64,
                                               reg_info[lldb_es_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_fs_x86_64,
                                               reg_info[lldb_fs_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_gs_x86_64,
                                               reg_info[lldb_gs_x86_64]));
  }

  if (*context_flags & ControlFlag) {
    writeRegister(source_data, getDestRegister(result_base, lldb_ss_x86_64,
                                               reg_info[lldb_ss_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_rflags_x86_64,
                                               reg_info[lldb_rflags_x86_64]));
  }

  if (*context_flags & DebugRegistersFlag) {
    source_data =
        source_data.drop_front(6 * 8); // 6 debug registers 64 bit each
  }

  if (*context_flags & IntegerFlag) {
    writeRegister(source_data, getDestRegister(result_base, lldb_rax_x86_64,
                                               reg_info[lldb_rax_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_rcx_x86_64,
                                               reg_info[lldb_rcx_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_rdx_x86_64,
                                               reg_info[lldb_rdx_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_rbx_x86_64,
                                               reg_info[lldb_rbx_x86_64]));
  }

  if (*context_flags & ControlFlag) {
    writeRegister(source_data, getDestRegister(result_base, lldb_rsp_x86_64,
                                               reg_info[lldb_rsp_x86_64]));
  }

  if (*context_flags & IntegerFlag) {
    writeRegister(source_data, getDestRegister(result_base, lldb_rbp_x86_64,
                                               reg_info[lldb_rbp_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_rsi_x86_64,
                                               reg_info[lldb_rsi_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_rdi_x86_64,
                                               reg_info[lldb_rdi_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_r8_x86_64,
                                               reg_info[lldb_r8_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_r9_x86_64,
                                               reg_info[lldb_r9_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_r10_x86_64,
                                               reg_info[lldb_r10_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_r11_x86_64,
                                               reg_info[lldb_r11_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_r12_x86_64,
                                               reg_info[lldb_r12_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_r13_x86_64,
                                               reg_info[lldb_r13_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_r14_x86_64,
                                               reg_info[lldb_r14_x86_64]));
    writeRegister(source_data, getDestRegister(result_base, lldb_r15_x86_64,
                                               reg_info[lldb_r15_x86_64]));
  }

  if (*context_flags & ControlFlag) {
    writeRegister(source_data, getDestRegister(result_base, lldb_rip_x86_64,
                                               reg_info[lldb_rip_x86_64]));
  }

  // TODO parse the floating point registers

  return result_context_buf;
}
