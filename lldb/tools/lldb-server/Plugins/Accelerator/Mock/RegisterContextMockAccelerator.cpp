//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextMockAccelerator.h"

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Register/RegisterValue.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;

// LLDB register numbers must start at 0 and be contiguous. This minimal set is
// just enough for the mock accelerator process to be debugged over GDB remote.
enum LLDBRegNum : uint32_t {
  LLDB_R0 = 0,
  LLDB_R1,
  LLDB_SP,
  LLDB_FP,
  LLDB_PC,
  LLDB_Flags,
  kNumRegs
};

#define DEFINE_REG(reg_id, name, idx, generic)                                 \
  RegisterDescriptor(reg_id, name, nullptr, sizeof(uint64_t), eEncodingUint,   \
                     eFormatHex,                                               \
                     {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, generic,       \
                      LLDB_INVALID_REGNUM, idx},                               \
                     {}, {}, nullptr)

static const RegisterDescriptor g_register_infos[] = {
    DEFINE_REG(LLDB_R0, "r0", LLDB_R0, LLDB_INVALID_REGNUM),
    DEFINE_REG(LLDB_R1, "r1", LLDB_R1, LLDB_INVALID_REGNUM),
    DEFINE_REG(LLDB_SP, "sp", LLDB_SP, LLDB_REGNUM_GENERIC_SP),
    DEFINE_REG(LLDB_FP, "fp", LLDB_FP, LLDB_REGNUM_GENERIC_FP),
    DEFINE_REG(LLDB_PC, "pc", LLDB_PC, LLDB_REGNUM_GENERIC_PC),
    DEFINE_REG(LLDB_Flags, "flags", LLDB_Flags, LLDB_INVALID_REGNUM)};

// The set's member register numbers. This must be a valid array (not null):
// the stop-reply path reads it to expedite the set's registers.
static const uint32_t g_register_nums[] = {LLDB_R0, LLDB_R1, LLDB_SP,
                                           LLDB_FP, LLDB_PC, LLDB_Flags};

static const RegisterGroup g_register_set = {
    "General Purpose Registers", "gpr",
    std::vector<RegisterID>(g_register_nums, g_register_nums + kNumRegs)};

static std::shared_ptr<const RegisterDescription> CreateDescription() {
  static const auto description = std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          {std::begin(g_register_infos), std::end(g_register_infos)},
          {g_register_set})));
  return description;
}

RegisterContextMockAccelerator::RegisterContextMockAccelerator(
    NativeThreadProtocol &native_thread)
    : NativeRegisterContext(native_thread, CreateDescription()) {
  // Give each register a distinct, constant value so reads are deterministic.
  for (uint32_t i = 0; i < kNumRegs; ++i)
    m_regs[i] = 0x1000 + i;
}

llvm::Expected<RegisterValue>
RegisterContextMockAccelerator::ReadRegister(RegisterID id) {
  if (id >= kNumRegs)
    return llvm::createStringError("invalid accelerator register");
  return RegisterValue(m_regs[id]);
}

llvm::Error
RegisterContextMockAccelerator::WriteRegister(RegisterID id,
                                              const RegisterValue &value) {
  if (id >= kNumRegs)
    return llvm::createStringError("invalid accelerator register");
  m_regs[id] = value.GetAsUInt64();
  return llvm::Error::success();
}

Status RegisterContextMockAccelerator::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(
      reinterpret_cast<const uint8_t *>(m_regs.data()),
      m_regs.size() * sizeof(uint64_t));
  return Status();
}

Status RegisterContextMockAccelerator::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (!data_sp || data_sp->GetByteSize() != m_regs.size() * sizeof(uint64_t))
    return Status::FromErrorString("invalid register data");
  ::memcpy(m_regs.data(), data_sp->GetBytes(),
           m_regs.size() * sizeof(uint64_t));
  return Status();
}
