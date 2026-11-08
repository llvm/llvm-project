//===-- RegisterContextMemory.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextMemory.h"

#include "lldb/Register/RegisterLayout.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

// RegisterContextMemory constructor
RegisterContextMemory::RegisterContextMemory(
    Thread &thread, uint32_t concrete_frame_idx,
    std::shared_ptr<const RegisterLayout> reg_infos_sp, addr_t reg_data_addr)
    : ThreadRegisterContext(thread, concrete_frame_idx),
      m_reg_infos_sp(std::move(reg_infos_sp)), m_reg_valid(), m_reg_data(),
      m_reg_data_addr(reg_data_addr) {
  SetRegisterDescription(m_reg_infos_sp->GetDescription());

  // Resize our vector of bools to contain one bool for every register. We will
  // use these boolean values to know when a register value is valid in
  // m_reg_data.
  const size_t num_regs =
      m_reg_infos_sp->GetDescription()->GetRegisters().size();
  assert(num_regs > 0);
  m_reg_valid.resize(num_regs);

  // Make a heap based buffer that is big enough to store all registers
  m_data = std::make_shared<DataBufferHeap>(m_reg_infos_sp->GetBankSize(0), 0);
  m_reg_data.SetData(m_data);
}

// Destructor
RegisterContextMemory::~RegisterContextMemory() = default;

void RegisterContextMemory::InvalidateAllRegisters() {
  if (m_reg_data_addr != LLDB_INVALID_ADDRESS)
    SetAllRegisterValid(false);
}

void RegisterContextMemory::SetAllRegisterValid(bool b) {
  std::vector<bool>::iterator pos, end = m_reg_valid.end();
  for (pos = m_reg_valid.begin(); pos != end; ++pos)
    *pos = b;
}

llvm::Expected<RegisterValue>
RegisterContextMemory::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  if (!reg_info)
    return llvm::createStringError("register value is unavailable");
  const uint32_t reg_num = reg_info->numbers[eRegisterKindLLDB];
  if (!m_reg_valid[reg_num]) {
    if (!ReadAllRegisterValues(m_data))
      return llvm::createStringError("register value is unavailable");
  }
  const auto *model = m_reg_infos_sp.get();
  const auto *reg =
      model ? model->GetDescription()->Find(reg_info->id) : nullptr;
  if (!reg)
    return llvm::createStringError("register value is unavailable");
  auto result = model->Read(
      reg->id, {{0, {m_reg_data.GetDataStart(), m_reg_data.GetByteSize()}}});
  return result;
}

llvm::Error
RegisterContextMemory::WriteRegister(RegisterID id,
                                     const RegisterValue &reg_value) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  const auto *model = m_reg_infos_sp.get();
  auto process = CalculateProcess();
  const auto *reg =
      model && reg_info ? model->GetDescription()->Find(reg_info->id) : nullptr;
  if (!reg || !process || m_reg_data_addr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("register write is unavailable");
  // This is a copy of the actual scripted memory-image format. Only the
  // mapped pieces are written back; unrelated register bytes are untouched.
  DataBufferHeap image(m_reg_data.GetDataStart(), m_reg_data.GetByteSize());
  if (auto error = model->Write(
          reg->id, reg_value, {{0, {image.GetBytes(), image.GetByteSize()}}})) {
    llvm::consumeError(std::move(error));
    return llvm::createStringError("register write is unavailable");
  }
  RegisterID written = reg->zero_extend_parent == LLDB_INVALID_REGNUM
                           ? reg->id
                           : reg->zero_extend_parent;
  const auto *location = model->Find(written);
  SetAllRegisterValid(false);
  for (const auto &piece : location->pieces) {
    Status error;
    if (piece.byte_offset > LLDB_INVALID_ADDRESS - m_reg_data_addr ||
        process->WriteMemory(m_reg_data_addr + piece.byte_offset,
                             image.GetBytes() + piece.byte_offset,
                             piece.byte_size, error) != piece.byte_size)
      return llvm::createStringError("register write is unavailable");
  }
  return llvm::Error::success();
}

bool RegisterContextMemory::ReadAllRegisterValues(
    WritableDataBufferSP &data_sp) {
  if (m_reg_data_addr != LLDB_INVALID_ADDRESS) {
    ProcessSP process_sp(CalculateProcess());
    if (process_sp) {
      Status error;
      if (process_sp->ReadMemory(m_reg_data_addr, data_sp->GetBytes(),
                                 data_sp->GetByteSize(),
                                 error) == data_sp->GetByteSize()) {
        SetAllRegisterValid(true);
        return true;
      }
    }
  }
  return false;
}

bool RegisterContextMemory::WriteAllRegisterValues(
    const DataBufferSP &data_sp) {
  if (m_reg_data_addr != LLDB_INVALID_ADDRESS) {
    ProcessSP process_sp(CalculateProcess());
    if (process_sp) {
      Status error;
      SetAllRegisterValid(false);
      if (process_sp->WriteMemory(m_reg_data_addr, data_sp->GetBytes(),
                                  data_sp->GetByteSize(),
                                  error) == data_sp->GetByteSize())
        return true;
    }
  }
  return false;
}

void RegisterContextMemory::SetAllRegisterData(
    const lldb::DataBufferSP &data_sp) {
  m_reg_data.SetData(data_sp);
  SetAllRegisterValid(true);
}
