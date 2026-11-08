//===-- RegisterContextWindows_x64.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__x86_64__) || defined(_M_X64)

#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-types.h"

#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "ProcessWindowsLog.h"
#include "RegisterLayoutWindows_x86.h"
#include "RegisterContextWindows_x64.h"
#include "TargetThreadWindows.h"
#include "lldb/Register/RegisterLayout.h"


namespace reg_ids = lldb_private::registers::x86_64::id;
using namespace lldb;
using namespace lldb_private;

namespace {
const RegisterLayout &RegisterMetadata() {
  static const ArchSpec arch("x86_64-pc-windows");
  static const auto info = CreateWindowsX86RegisterLayout(
      arch, X86RegisterMode::X86_64, WindowsX86ContextKind::Frame);
  return info;
}
} // namespace

// Constructors and Destructors
RegisterContextWindows_x64::RegisterContextWindows_x64(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx) {
  SetRegisterLayout(RegisterMetadata());
}

RegisterContextWindows_x64::~RegisterContextWindows_x64() {}

llvm::Expected<RegisterValue>
RegisterContextWindows_x64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue reg_value;
  if (!CacheAllRegisterValues())
    return llvm::createStringError("register value is unavailable");

  if (reg_info == nullptr)
    return llvm::createStringError("register value is unavailable");

  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];

  switch (reg) {
#define GPR_CASE(size, reg_case, reg_val)                                      \
  case reg_case:                                                               \
    reg_value.SetUInt##size(reg_val);                                        \
    break;

    GPR_CASE(64, reg_ids::rax, m_context.Rax);
    GPR_CASE(64, reg_ids::rbx, m_context.Rbx);
    GPR_CASE(64, reg_ids::rcx, m_context.Rcx);
    GPR_CASE(64, reg_ids::rdx, m_context.Rdx);
    GPR_CASE(64, reg_ids::rdi, m_context.Rdi);
    GPR_CASE(64, reg_ids::rsi, m_context.Rsi);
    GPR_CASE(64, reg_ids::r8, m_context.R8);
    GPR_CASE(64, reg_ids::r9, m_context.R9);
    GPR_CASE(64, reg_ids::r10, m_context.R10);
    GPR_CASE(64, reg_ids::r11, m_context.R11);
    GPR_CASE(64, reg_ids::r12, m_context.R12);
    GPR_CASE(64, reg_ids::r13, m_context.R13);
    GPR_CASE(64, reg_ids::r14, m_context.R14);
    GPR_CASE(64, reg_ids::r15, m_context.R15);
    GPR_CASE(64, reg_ids::rbp, m_context.Rbp);
    GPR_CASE(64, reg_ids::rsp, m_context.Rsp);
    GPR_CASE(64, reg_ids::rip, m_context.Rip);
    GPR_CASE(64, reg_ids::rflags, m_context.EFlags);
    GPR_CASE(32, reg_ids::eax, static_cast<uint32_t>(m_context.Rax));
    GPR_CASE(32, reg_ids::ebx, static_cast<uint32_t>(m_context.Rbx));
    GPR_CASE(32, reg_ids::ecx, static_cast<uint32_t>(m_context.Rcx));
    GPR_CASE(32, reg_ids::edx, static_cast<uint32_t>(m_context.Rdx));
    GPR_CASE(32, reg_ids::edi, static_cast<uint32_t>(m_context.Rdi));
    GPR_CASE(32, reg_ids::esi, static_cast<uint32_t>(m_context.Rsi));
    GPR_CASE(32, reg_ids::ebp, static_cast<uint32_t>(m_context.Rbp));
    GPR_CASE(32, reg_ids::esp, static_cast<uint32_t>(m_context.Rsp));
    GPR_CASE(32, reg_ids::r8d, static_cast<uint32_t>(m_context.R8));
    GPR_CASE(32, reg_ids::r9d, static_cast<uint32_t>(m_context.R9));
    GPR_CASE(32, reg_ids::r10d, static_cast<uint32_t>(m_context.R10));
    GPR_CASE(32, reg_ids::r11d, static_cast<uint32_t>(m_context.R11));
    GPR_CASE(32, reg_ids::r12d, static_cast<uint32_t>(m_context.R12));
    GPR_CASE(32, reg_ids::r13d, static_cast<uint32_t>(m_context.R13));
    GPR_CASE(32, reg_ids::r14d, static_cast<uint32_t>(m_context.R14));
    GPR_CASE(32, reg_ids::r15d, static_cast<uint32_t>(m_context.R15));
    GPR_CASE(16, reg_ids::ax, static_cast<uint16_t>(m_context.Rax));
    GPR_CASE(16, reg_ids::bx, static_cast<uint16_t>(m_context.Rbx));
    GPR_CASE(16, reg_ids::cx, static_cast<uint16_t>(m_context.Rcx));
    GPR_CASE(16, reg_ids::dx, static_cast<uint16_t>(m_context.Rdx));
    GPR_CASE(16, reg_ids::di, static_cast<uint16_t>(m_context.Rdi));
    GPR_CASE(16, reg_ids::si, static_cast<uint16_t>(m_context.Rsi));
    GPR_CASE(16, reg_ids::bp, static_cast<uint16_t>(m_context.Rbp));
    GPR_CASE(16, reg_ids::sp, static_cast<uint16_t>(m_context.Rsp));
    GPR_CASE(16, reg_ids::r8w, static_cast<uint16_t>(m_context.R8));
    GPR_CASE(16, reg_ids::r9w, static_cast<uint16_t>(m_context.R9));
    GPR_CASE(16, reg_ids::r10w, static_cast<uint16_t>(m_context.R10));
    GPR_CASE(16, reg_ids::r11w, static_cast<uint16_t>(m_context.R11));
    GPR_CASE(16, reg_ids::r12w, static_cast<uint16_t>(m_context.R12));
    GPR_CASE(16, reg_ids::r13w, static_cast<uint16_t>(m_context.R13));
    GPR_CASE(16, reg_ids::r14w, static_cast<uint16_t>(m_context.R14));
    GPR_CASE(16, reg_ids::r15w, static_cast<uint16_t>(m_context.R15));
    GPR_CASE(8, reg_ids::ah, static_cast<uint16_t>(m_context.Rax) >> 8);
    GPR_CASE(8, reg_ids::bh, static_cast<uint16_t>(m_context.Rbx) >> 8);
    GPR_CASE(8, reg_ids::ch, static_cast<uint16_t>(m_context.Rcx) >> 8);
    GPR_CASE(8, reg_ids::dh, static_cast<uint16_t>(m_context.Rdx) >> 8);
    GPR_CASE(8, reg_ids::al, static_cast<uint8_t>(m_context.Rax));
    GPR_CASE(8, reg_ids::bl, static_cast<uint8_t>(m_context.Rbx));
    GPR_CASE(8, reg_ids::cl, static_cast<uint8_t>(m_context.Rcx));
    GPR_CASE(8, reg_ids::dl, static_cast<uint8_t>(m_context.Rdx));
    GPR_CASE(8, reg_ids::dil, static_cast<uint8_t>(m_context.Rdi));
    GPR_CASE(8, reg_ids::sil, static_cast<uint8_t>(m_context.Rsi));
    GPR_CASE(8, reg_ids::bpl, static_cast<uint8_t>(m_context.Rbp));
    GPR_CASE(8, reg_ids::spl, static_cast<uint8_t>(m_context.Rsp));
    GPR_CASE(8, reg_ids::r8l, static_cast<uint8_t>(m_context.R8));
    GPR_CASE(8, reg_ids::r9l, static_cast<uint8_t>(m_context.R9));
    GPR_CASE(8, reg_ids::r10l, static_cast<uint8_t>(m_context.R10));
    GPR_CASE(8, reg_ids::r11l, static_cast<uint8_t>(m_context.R11));
    GPR_CASE(8, reg_ids::r12l, static_cast<uint8_t>(m_context.R12));
    GPR_CASE(8, reg_ids::r13l, static_cast<uint8_t>(m_context.R13));
    GPR_CASE(8, reg_ids::r14l, static_cast<uint8_t>(m_context.R14));
    GPR_CASE(8, reg_ids::r15l, static_cast<uint8_t>(m_context.R15));

  case reg_ids::xmm0:
    reg_value.SetBytes(&m_context.Xmm0,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm1:
    reg_value.SetBytes(&m_context.Xmm1,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm2:
    reg_value.SetBytes(&m_context.Xmm2,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm3:
    reg_value.SetBytes(&m_context.Xmm3,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm4:
    reg_value.SetBytes(&m_context.Xmm4,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm5:
    reg_value.SetBytes(&m_context.Xmm5,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm6:
    reg_value.SetBytes(&m_context.Xmm6,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm7:
    reg_value.SetBytes(&m_context.Xmm7,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm8:
    reg_value.SetBytes(&m_context.Xmm8,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm9:
    reg_value.SetBytes(&m_context.Xmm9,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm10:
    reg_value.SetBytes(&m_context.Xmm10,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm11:
    reg_value.SetBytes(&m_context.Xmm11,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm12:
    reg_value.SetBytes(&m_context.Xmm12,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm13:
    reg_value.SetBytes(&m_context.Xmm13,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm14:
    reg_value.SetBytes(&m_context.Xmm14,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  case reg_ids::xmm15:
    reg_value.SetBytes(&m_context.Xmm15,
                       reg_info->byte_size, endian::InlHostByteOrder());
    break;
  }
  return reg_value;
}

llvm::Error
RegisterContextWindows_x64::WriteRegister(RegisterID id,
                                               const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  // Since we cannot only write a single register value to the inferior, we
  // need to make sure our cached copy of the register values are fresh.
  // Otherwise when writing EAX, for example, we may also overwrite some other
  // register with a stale value.
  if (!CacheAllRegisterValues())
    return llvm::createStringError("register write is unavailable");

  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];

  Log *log = GetLog(WindowsLog::Registers);
  switch (reg) {
  case reg_ids::rax:
    m_context.Rax = reg_value.GetAsUInt64();
    break;
  case reg_ids::rbx:
    m_context.Rbx = reg_value.GetAsUInt64();
    break;
  case reg_ids::rcx:
    m_context.Rcx = reg_value.GetAsUInt64();
    break;
  case reg_ids::rdx:
    m_context.Rdx = reg_value.GetAsUInt64();
    break;
  case reg_ids::rdi:
    m_context.Rdi = reg_value.GetAsUInt64();
    break;
  case reg_ids::rsi:
    m_context.Rsi = reg_value.GetAsUInt64();
    break;
  case reg_ids::r8:
    m_context.R8 = reg_value.GetAsUInt64();
    break;
  case reg_ids::r9:
    m_context.R9 = reg_value.GetAsUInt64();
    break;
  case reg_ids::r10:
    m_context.R10 = reg_value.GetAsUInt64();
    break;
  case reg_ids::r11:
    m_context.R11 = reg_value.GetAsUInt64();
    break;
  case reg_ids::r12:
    m_context.R12 = reg_value.GetAsUInt64();
    break;
  case reg_ids::r13:
    m_context.R13 = reg_value.GetAsUInt64();
    break;
  case reg_ids::r14:
    m_context.R14 = reg_value.GetAsUInt64();
    break;
  case reg_ids::r15:
    m_context.R15 = reg_value.GetAsUInt64();
    break;
  case reg_ids::rbp:
    m_context.Rbp = reg_value.GetAsUInt64();
    break;
  case reg_ids::rsp:
    m_context.Rsp = reg_value.GetAsUInt64();
    break;
  case reg_ids::rip:
    m_context.Rip = reg_value.GetAsUInt64();
    break;
  case reg_ids::rflags:
    m_context.EFlags = reg_value.GetAsUInt64();
    break;
  case reg_ids::xmm0:
    memcpy(&m_context.Xmm0, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm1:
    memcpy(&m_context.Xmm1, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm2:
    memcpy(&m_context.Xmm2, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm3:
    memcpy(&m_context.Xmm3, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm4:
    memcpy(&m_context.Xmm4, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm5:
    memcpy(&m_context.Xmm5, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm6:
    memcpy(&m_context.Xmm6, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm7:
    memcpy(&m_context.Xmm7, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm8:
    memcpy(&m_context.Xmm8, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm9:
    memcpy(&m_context.Xmm9, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm10:
    memcpy(&m_context.Xmm10, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm11:
    memcpy(&m_context.Xmm11, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm12:
    memcpy(&m_context.Xmm12, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm13:
    memcpy(&m_context.Xmm13, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm14:
    memcpy(&m_context.Xmm14, reg_value.GetBytes(), 16);
    break;
  case reg_ids::xmm15:
    memcpy(&m_context.Xmm15, reg_value.GetBytes(), 16);
    break;
  default:
    LLDB_LOG(log, "Write value {0:x} to unknown register {1}",
             reg_value.GetAsUInt32(), reg);
    return llvm::createStringError("register write is unavailable");
  }

  // Physically update the registers in the target process.
  {
    if (!(ApplyAllRegisterValues()))
      return llvm::createStringError("register write failed");
    return llvm::Error::success();
  }
}

#endif // defined(__x86_64__) || defined(_M_X64)
