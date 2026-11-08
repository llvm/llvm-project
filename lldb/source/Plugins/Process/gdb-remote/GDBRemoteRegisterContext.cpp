//===-- GDBRemoteRegisterContext.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GDBRemoteRegisterContext.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"
#include "ThreadGDBRemote.h"
#include "lldb/Register/Arch/lldb-arm-dwarf-register-enums.h"
#include "lldb/Core/Architecture.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StringExtractorGDBRemote.h"

#include <functional>
#include <memory>
#include <set>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

// GDBRemoteRegisterContext constructor
GDBRemoteRegisterContext::GDBRemoteRegisterContext(
    ThreadGDBRemote &thread, uint32_t concrete_frame_idx,
    std::shared_ptr<const RegisterLayout> layout_sp, bool read_all_at_once,
    bool write_all_at_once)
    : ThreadRegisterContext(thread, concrete_frame_idx),
      m_register_layout_sp(std::move(layout_sp)), m_reg_valid(), m_reg_data(),
      m_read_all_at_once(read_all_at_once),
      m_write_all_at_once(write_all_at_once), m_gpacket_cached(false) {
  SetRegisterDescription(m_register_layout_sp->GetDescription());

  // Resize our vector of bools to contain one bool for every register. We will
  // use these boolean values to know when a register value is valid in
  // m_reg_data.
  m_reg_valid.resize(
      m_register_layout_sp->GetDescription()->GetRegisters().size());
  SetAllRegistersValidState(eLazyBoolCalculate);

  // Make a heap based buffer that is big enough to store all registers
  DataBufferSP reg_data_sp(
      new DataBufferHeap(m_register_layout_sp->GetBankSize(0), 0));
  m_reg_data.SetData(reg_data_sp);
  m_reg_data.SetByteOrder(thread.GetProcess()->GetByteOrder());
}

// Destructor
GDBRemoteRegisterContext::~GDBRemoteRegisterContext() = default;

void GDBRemoteRegisterContext::SetAllRegistersValidState(LazyBool valid) {
  m_gpacket_cached = (valid == eLazyBoolYes);
  for (auto &reg : m_reg_valid)
    reg = valid;
}

LazyBool GDBRemoteRegisterContext::GetRegisterValidState(uint32_t reg) const {
  assert(reg < m_reg_valid.size());
  if (reg < m_reg_valid.size())
    return m_reg_valid[reg];
  return eLazyBoolNo;
}

void GDBRemoteRegisterContext::SetRegisterIsValidState(
    const RegisterDescriptor *reg_info, LazyBool valid) {
  if (reg_info)
    SetRegisterIsValidState(
        *m_register_layout_sp->GetDescription()->GetRegisterIndex(reg_info->id),
        valid);
}

void GDBRemoteRegisterContext::SetRegisterIsValidState(uint32_t reg,
                                                       LazyBool valid) {
  assert(reg < m_reg_valid.size());
  if (reg < m_reg_valid.size())
    m_reg_valid[reg] = valid;
}

llvm::Expected<RegisterValue>
GDBRemoteRegisterContext::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  if (!reg_info || !ReadRegisterBytes(reg_info) ||
      GetRegisterIsUnavailable(
          *m_register_layout_sp->GetDescription()->GetRegisterIndex(
              reg_info->id)))
    return llvm::createStringError("register value is unavailable");
  const auto *layout = m_register_layout_sp.get();
  const auto *reg =
      layout ? layout->GetDescription()->Find(reg_info->id) : nullptr;
  if (!reg)
    return llvm::createStringError("register value is unavailable");
  auto result = (*layout).Read(
      reg->id, {{0, {m_reg_data.GetDataStart(), m_reg_data.GetByteSize()}}});
  return result;
}

bool GDBRemoteRegisterContext::PrivateSetRegisterValue(
    RegisterID id, llvm::ArrayRef<uint8_t> data) {
  const auto *layout = m_register_layout_sp.get();
  const auto *descriptor =
      layout ? layout->GetDescription()->Find(id) : nullptr;
  if (!descriptor || m_reg_data.GetByteOrder() == eByteOrderInvalid)
    return false;
  InvalidateIfNeeded(false);
  const auto &reg = *descriptor;
  uint32_t index = *layout->GetDescription()->GetRegisterIndex(id);
  if (data.size() < reg.byte_size) {
    if (!data.empty())
      SetRegisterIsUnfetched(index);
    return false;
  }
  RegisterValue value(data.take_front(reg.byte_size),
                      m_reg_data.GetByteOrder());
  if (auto error =
          (*layout).Write(reg.id, value,
                          {{0,
                            {const_cast<uint8_t *>(m_reg_data.GetDataStart()),
                             m_reg_data.GetByteSize()}}})) {
    llvm::consumeError(std::move(error));
    return false;
  }
  SetRegisterIsValid(index);
  return true;
}

bool GDBRemoteRegisterContext::PrivateSetRegisterValue(RegisterID id,
                                                       uint64_t value) {
  const RegisterDescriptor *reg = GetDescription()->Find(id);
  if (!reg || m_reg_data.GetByteOrder() == eByteOrderInvalid)
    return false;
  DataExtractor encoded;
  if (!RegisterValue(value).GetData(encoded, reg->byte_size,
                                    m_reg_data.GetByteOrder()))
    return false;
  return PrivateSetRegisterValue(
      id, {encoded.GetDataStart(), encoded.GetByteSize()});
}

// Helper function for GDBRemoteRegisterContext::ReadRegisterBytes().
bool GDBRemoteRegisterContext::GetPrimordialRegister(
    const RegisterDescriptor *reg_info,
    GDBRemoteCommunicationClient &gdb_comm) {
  const uint32_t remote_reg = reg_info->numbers[eRegisterKindProcessPlugin];

  if (DataBufferSP buffer_sp =
          gdb_comm.ReadRegister(m_thread.GetProtocolID(), remote_reg))
    return PrivateSetRegisterValue(
        reg_info->id, llvm::ArrayRef<uint8_t>(buffer_sp->GetBytes(),
                                          buffer_sp->GetByteSize()));
  return false;
}

bool GDBRemoteRegisterContext::ReadRegisterBytes(
    const RegisterDescriptor *reg_info) {
  ExecutionContext exe_ctx(CalculateThread());

  Process *process = exe_ctx.GetProcessPtr();
  Thread *thread = exe_ctx.GetThreadPtr();
  if (process == nullptr || thread == nullptr)
    return false;

  GDBRemoteCommunicationClient &gdb_comm(
      ((ProcessGDBRemote *)process)->GetGDBRemote());

  InvalidateIfNeeded(false);

  const uint32_t reg =
      *m_register_layout_sp->GetDescription()->GetRegisterIndex(reg_info->id);

  if (GetRegisterIsUnfetched(reg)) {
    if (m_read_all_at_once && !m_gpacket_cached) {
      if (DataBufferSP buffer_sp =
              gdb_comm.ReadAllRegisters(m_thread.GetProtocolID())) {
        memcpy(const_cast<uint8_t *>(m_reg_data.GetDataStart()),
               buffer_sp->GetBytes(),
               std::min(buffer_sp->GetByteSize(), m_reg_data.GetByteSize()));
        if (buffer_sp->GetByteSize() >= m_reg_data.GetByteSize()) {
          SetAllRegistersValid();
          return true;
        } else if (buffer_sp->GetByteSize() > 0) {
          for (size_t i = 0;
               i <
               m_register_layout_sp->GetDescription()->GetRegisters().size();
               ++i) {
            const auto &reginfo =
                *m_register_layout_sp->GetDescription()->GetRegisterAtIndex(i);
            uint64_t offset =
                GetRegisterImageOffset(*m_register_layout_sp, reginfo.id);
            if (offset != LLDB_INVALID_INDEX32 &&
                offset + reginfo.byte_size <= buffer_sp->GetByteSize())
              SetRegisterIsValid(i);
            else
              SetRegisterIsUnfetched(i);
          }

          m_gpacket_cached = true;
          if (GetRegisterIsValid(reg))
            return true;
        } else {
          Log *log(GetLog(GDBRLog::Thread | GDBRLog::Packets));
          LLDB_LOGF(
              log,
              "error: GDBRemoteRegisterContext::ReadRegisterBytes tried "
              "to read the "
              "entire register context at once, expected at least %" PRId64
              " bytes "
              "but only got %" PRId64 " bytes.",
              m_reg_data.GetByteSize(), buffer_sp->GetByteSize());
          return false;
        }
      }
    }
    if (!reg_info->value_registers.empty()) {
      // Process this composite register request by delegating to the
      // constituent primordial registers.

      bool success = true;
      for (RegisterID parent : reg_info->value_registers) {
        const auto *prim_reg_info = GetDescription()->Find(parent);
        if (!prim_reg_info) {
          success = false;
          break;
        }
        const size_t prim_reg = *GetDescription()->GetRegisterIndex(parent);
        if (GetRegisterIsUnfetched(prim_reg))
          success = ReadRegisterBytes(prim_reg_info);
        else if (GetRegisterIsUnavailable(prim_reg))
          success = false;
        if (!success)
          break;
      }

      if (success) {
        // If we reach this point, all primordial register requests have
        // succeeded. Validate this composite register.
        SetRegisterIsValid(reg_info);
      } else {
        SetRegisterIsUnfetched(reg_info);
      }
    } else {
      // Get each register individually
      if (GetPrimordialRegister(reg_info, gdb_comm))
        SetRegisterIsValid(reg_info);
      else
        SetRegisterIsUnfetched(reg_info);
    }

    // Make sure we got a valid register value after reading it
    if (GetRegisterIsUnfetched(reg) || GetRegisterIsUnavailable(reg))
      return false;
  }

  return true;
}

llvm::Error
GDBRemoteRegisterContext::WriteRegister(RegisterID id,
                                             const RegisterValue &value) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  if (!reg_info || !m_register_layout_sp.get())
    return llvm::createStringError("register write is unavailable");
  auto process = std::static_pointer_cast<ProcessGDBRemote>(CalculateProcess());
  if (!process)
    return llvm::createStringError("register write is unavailable");
  InvalidateIfNeeded(false);
  // Retain this exact metadata generation throughout encoding and transport.
  RegisterLayout layout = *m_register_layout_sp.get();
  const auto *reg = layout.GetDescription()->Find(reg_info->id);
  if (!reg)
    return llvm::createStringError("register write is unavailable");
  const bool reconfigure = RegisterWriteCausesReconfigure(reg->name);
  std::set<RegisterID> primary;
  std::function<void(RegisterID)> collect = [&](RegisterID id) {
    const auto &value = *layout.GetDescription()->Find(id);
    if (value.value_registers.empty())
      primary.insert(id);
    else
      for (RegisterID parent : value.value_registers)
        collect(parent);
  };
  collect(reg->id);

  // Preserve the remaining bytes of aliased parents. G writes use the cached
  // image, which can include slots the stub does not allow us to read.
  auto fetch = [&](RegisterID id) {
    return ReadRegisterBytes(layout.GetDescription()->Find(id));
  };
  if (!reg->value_registers.empty()) {
    for (RegisterID id : primary)
      if (!fetch(id))
        return llvm::createStringError("register write is unavailable");
  }

  auto &communication = process->GetGDBRemote();
  GDBRemoteClientBase::Lock lock(communication);
  if (!lock)
    return llvm::createStringError("register write is unavailable");
  auto bytes = llvm::MutableArrayRef<uint8_t>(
      const_cast<uint8_t *>(m_reg_data.GetDataStart()),
      m_reg_data.GetByteSize());
  if (auto error = layout.Write(reg->id, value, {{0, bytes}})) {
    llvm::consumeError(std::move(error));
    return llvm::createStringError("register write is unavailable");
  }
  bool success = true;
  if (m_write_all_at_once) {
    success = communication.WriteAllRegisters(m_thread.GetProtocolID(), bytes);
    InvalidateAllRegisters();
  } else {
    for (RegisterID id : primary) {
      const auto *info = layout.GetDescription()->Find(id);
      if (!SetPrimordialRegister(info, communication)) {
        success = false;
        break;
      }
    }
    SetRegisterIsUnfetched(reg_info);
    for (RegisterID id : reg->invalidate_registers)
      SetRegisterIsUnfetched(*layout.GetDescription()->GetRegisterIndex(id));
  }
  if (!success) {
    // A failed packet sequence can have written some parents already.
    InvalidateAllRegisters();
    return llvm::createStringError("register write is unavailable");
  }
  if (reconfigure && ReconfigureRegisterLayout())
    InvalidateAllRegisters();
  return llvm::Error::success();
}

// Send one primary value from the real packet image.
bool GDBRemoteRegisterContext::SetPrimordialRegister(
    const RegisterDescriptor *reg_info,
    GDBRemoteCommunicationClient &gdb_comm) {
  const uint32_t reg =
      *m_register_layout_sp->GetDescription()->GetRegisterIndex(reg_info->id);
  // Invalidate just this register
  SetRegisterIsUnfetched(reg);

  return gdb_comm.WriteRegister(
      m_thread.GetProtocolID(), reg_info->numbers[eRegisterKindProcessPlugin],
      {m_reg_data.PeekData(
           GetRegisterImageOffset(*m_register_layout_sp, reg_info->id),
           reg_info->byte_size),
       reg_info->byte_size});
}

llvm::Expected<RegisterCheckpoint> GDBRemoteRegisterContext::SaveCheckpoint() {
  auto process = std::static_pointer_cast<ProcessGDBRemote>(CalculateProcess());
  if (!process)
    return llvm::createStringError("process is unavailable");
  uint32_t save_id = 0;
  if (!process->GetGDBRemote().SaveRegisterState(m_thread.GetProtocolID(),
                                                 save_id)) {
    WritableDataBufferSP data;
    if (!ReadAllRegisterValues(data))
      return llvm::createStringError(
          "could not capture remote register values");
    auto layout = m_register_layout_sp;
    auto description = GetDescription();
    std::vector<std::pair<RegisterID, RegisterValue>> controls;
    for (const auto &reg : layout->GetDescription()->GetRegisters()) {
      if (!RegisterWriteCausesReconfigure(reg.name))
        continue;
      auto value =
          layout->Read(reg.id, {{0, {data->GetBytes(), data->GetByteSize()}}});
      if (value)
        controls.emplace_back(reg.id, std::move(*value));
      else
        llvm::consumeError(value.takeError());
    }
    return CreateCheckpoint(
        [source = std::static_pointer_cast<GDBRemoteRegisterContext>(
             shared_from_this()),
         data = std::move(data), layout, description,
         controls = std::move(controls)]() -> llvm::Error {
          auto &context = *source;
          if (!layout->HasSameStorage(*context.m_register_layout_sp))
            for (const auto &[id, value] : controls)
              if (auto error = context.WriteRegister(id, value))
                return error;
          if (!layout->HasSameStorage(*context.m_register_layout_sp))
            return llvm::createStringError(
                "remote register format changed since checkpoint");
          if (!context.WriteAllRegisterValues(data))
            return llvm::createStringError(
                "could not restore remote register values");
          context.SetRegisterDescription(description);
          return llvm::Error::success();
        });
  }
  auto layout = m_register_layout_sp;
  auto description = GetDescription();
  return CreateCheckpoint(
      [source = std::static_pointer_cast<GDBRemoteRegisterContext>(
           shared_from_this()),
       save_id, layout, description]() -> llvm::Error {
        auto &context = *source;
        auto process = std::static_pointer_cast<ProcessGDBRemote>(
            context.CalculateProcess());
        if (!process || !process->GetGDBRemote().RestoreRegisterState(
                            context.m_thread.GetProtocolID(), save_id))
          return llvm::createStringError(
              "could not restore remote register state");
        context.m_register_layout_sp = layout;
        context.SetRegisterDescription(description);
        context.m_reg_valid.resize(
            layout->GetDescription()->GetRegisters().size());
        context.m_reg_data.SetData(
            std::make_shared<DataBufferHeap>(layout->GetBankSize(0), 0));
        context.InvalidateAllRegisters();
        return llvm::Error::success();
      });
}

bool GDBRemoteRegisterContext::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  ExecutionContext exe_ctx(CalculateThread());

  Process *process = exe_ctx.GetProcessPtr();
  Thread *thread = exe_ctx.GetThreadPtr();
  if (process == nullptr || thread == nullptr)
    return false;

  GDBRemoteCommunicationClient &gdb_comm(
      ((ProcessGDBRemote *)process)->GetGDBRemote());

  const bool use_g_packet =
      !gdb_comm.AvoidGPackets((ProcessGDBRemote *)process);

  GDBRemoteClientBase::Lock lock(gdb_comm);
  if (lock) {
    if (gdb_comm.SyncThreadState(m_thread.GetProtocolID()))
      InvalidateAllRegisters();

    if (use_g_packet) {
      if (DataBufferSP data_buffer =
              gdb_comm.ReadAllRegisters(m_thread.GetProtocolID())) {
        data_sp = std::make_shared<DataBufferHeap>(*data_buffer);
        return true;
      }
    }

    // We're going to read each register
    // individually and store them as binary data in a buffer.
    const RegisterDescriptor *reg_info;

    for (uint32_t i = 0; (reg_info = GetRegisterAtIndex(i)) != nullptr;
         i++) {
      if (!reg_info->value_registers
               .empty()) // skip registers that are slices of real registers
        continue;
      ReadRegisterBytes(reg_info);
      // ReadRegisterBytes saves the contents of the register in to the
      // m_reg_data buffer
    }
    data_sp = std::make_shared<DataBufferHeap>(
        m_reg_data.GetDataStart(), m_register_layout_sp->GetBankSize(0));
    return true;
  } else {

    Log *log(GetLog(GDBRLog::Thread | GDBRLog::Packets));
    if (log) {
      if (log->GetVerbose()) {
        StreamString strm;
        process->DumpPluginHistory(strm);
        LLDB_LOGF(log,
                  "error: failed to get packet sequence mutex, not sending "
                  "read all registers:\n%s",
                  strm.GetData());
      } else
        LLDB_LOGF(log,
                  "error: failed to get packet sequence mutex, not sending "
                  "read all registers");
    }
  }

  data_sp.reset();
  return false;
}

bool GDBRemoteRegisterContext::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (!data_sp || data_sp->GetBytes() == nullptr || data_sp->GetByteSize() == 0)
    return false;

  ExecutionContext exe_ctx(CalculateThread());

  Process *process = exe_ctx.GetProcessPtr();
  Thread *thread = exe_ctx.GetThreadPtr();
  if (process == nullptr || thread == nullptr)
    return false;

  GDBRemoteCommunicationClient &gdb_comm(
      ((ProcessGDBRemote *)process)->GetGDBRemote());

  const bool use_g_packet =
      !gdb_comm.AvoidGPackets((ProcessGDBRemote *)process);

  GDBRemoteClientBase::Lock lock(gdb_comm);
  if (lock) {
    // The data_sp contains the G response packet.
    if (use_g_packet) {
      if (gdb_comm.WriteAllRegisters(
              m_thread.GetProtocolID(),
              {data_sp->GetBytes(), size_t(data_sp->GetByteSize())}))
        return true;

      uint32_t num_restored = 0;
      // We need to manually go through all of the registers and restore them
      // manually
      DataExtractor restore_data(data_sp, m_reg_data.GetByteOrder(),
                                 m_reg_data.GetAddressByteSize());

      const RegisterDescriptor *reg_info;

      // The g packet contents may either include the slice registers
      // (registers defined in terms of other registers, e.g. eax is a subset
      // of rax) or not.  The slice registers should NOT be in the g packet,
      // but some implementations may incorrectly include them.
      //
      // If the slice registers are included in the packet, we must step over
      // the slice registers when parsing the packet -- relying on the
      // RegisterDescriptor byte_offset field would be incorrect. If the slice
      // registers are not included, then using the byte_offset values into the
      // data buffer is the best way to find individual register values.

      uint64_t size_including_slice_registers = 0;
      uint64_t size_not_including_slice_registers = 0;
      uint64_t size_by_highest_offset = 0;

      for (uint32_t reg_idx = 0;
           (reg_info = GetRegisterAtIndex(reg_idx)) != nullptr; ++reg_idx) {
        size_including_slice_registers += reg_info->byte_size;
        if (reg_info->value_registers.empty())
          size_not_including_slice_registers += reg_info->byte_size;
        uint64_t offset =
            GetRegisterImageOffset(*m_register_layout_sp, reg_info->id);
        if (offset != LLDB_INVALID_INDEX32)
          size_by_highest_offset =
              std::max(size_by_highest_offset, offset + reg_info->byte_size);
      }

      bool use_byte_offset_into_buffer;
      if (size_by_highest_offset == restore_data.GetByteSize()) {
        // The size of the packet agrees with the highest offset: + size in the
        // register file
        use_byte_offset_into_buffer = true;
      } else if (size_not_including_slice_registers ==
                 restore_data.GetByteSize()) {
        // The size of the packet is the same as concatenating all of the
        // registers sequentially, skipping the slice registers
        use_byte_offset_into_buffer = true;
      } else if (size_including_slice_registers == restore_data.GetByteSize()) {
        // The slice registers are present in the packet (when they shouldn't
        // be). Don't try to use the RegisterDescriptor byte_offset into the
        // restore_data, it will point to the wrong place.
        use_byte_offset_into_buffer = false;
      } else {
        // None of our expected sizes match the actual g packet data we're
        // looking at. The most conservative approach here is to use the
        // running total byte offset.
        use_byte_offset_into_buffer = false;
      }

      // In case our register definitions don't include the correct offsets,
      // keep track of the size of each reg & compute offset based on that.
      uint32_t running_byte_offset = 0;
      for (uint32_t reg_idx = 0;
           (reg_info = GetRegisterAtIndex(reg_idx)) != nullptr;
           ++reg_idx, running_byte_offset += reg_info->byte_size) {
        // Skip composite aka slice registers (e.g. eax is a slice of rax).
        if (!reg_info->value_registers.empty())
          continue;

        const uint32_t reg =
            *m_register_layout_sp->GetDescription()->GetRegisterIndex(
                reg_info->id);

        uint32_t register_offset;
        if (use_byte_offset_into_buffer) {
          register_offset =
              GetRegisterImageOffset(*m_register_layout_sp, reg_info->id);
        } else {
          register_offset = running_byte_offset;
        }

        const uint32_t reg_byte_size = reg_info->byte_size;

        const uint8_t *restore_src =
            restore_data.PeekData(register_offset, reg_byte_size);
        if (restore_src) {
          SetRegisterIsUnfetched(reg);
          if (gdb_comm.WriteRegister(
                  m_thread.GetProtocolID(),
                  reg_info->numbers[eRegisterKindProcessPlugin],
                  {restore_src, reg_byte_size}))
            ++num_restored;
        }
      }
      return num_restored > 0;
    } else {
      // For the use_g_packet == false case, we're going to write each register
      // individually.  The data buffer is binary data in this case, instead of
      // ascii characters.

      // The buffer is only valid for the register layout it was saved with.
      if (data_sp->GetByteSize() != m_register_layout_sp->GetBankSize(0))
        return false;

      bool arm64_debugserver = false;
      if (m_thread.GetProcess().get()) {
        const ArchSpec &arch =
            m_thread.GetProcess()->GetTarget().GetArchitecture();
        if (arch.IsValid() && (arch.GetMachine() == llvm::Triple::aarch64 ||
                               arch.GetMachine() == llvm::Triple::aarch64_32) &&
            arch.GetTriple().getVendor() == llvm::Triple::Apple &&
            arch.GetTriple().getOS() == llvm::Triple::IOS) {
          arm64_debugserver = true;
        }
      }
      uint32_t num_restored = 0;
      const RegisterDescriptor *reg_info;
      for (uint32_t i = 0; (reg_info = GetRegisterAtIndex(i)) != nullptr;
           i++) {
        if (!reg_info->value_registers.empty()) // skip registers that are
                                                // slices of real registers
          continue;
        // Skip the fpsr and fpcr floating point status/control register
        // writing to work around a bug in an older version of debugserver that
        // would lead to register context corruption when writing fpsr/fpcr.
        if (arm64_debugserver &&
            (strcmp(reg_info->name.c_str(), "fpsr") == 0 ||
             strcmp(reg_info->name.c_str(), "fpcr") == 0)) {
          continue;
        }

        SetRegisterIsUnfetched(reg_info);
        if (gdb_comm.WriteRegister(
                m_thread.GetProtocolID(),
                reg_info->numbers[eRegisterKindProcessPlugin],
                {data_sp->GetBytes() + GetRegisterImageOffset(
                                           *m_register_layout_sp, reg_info->id),
                 reg_info->byte_size}))
          ++num_restored;
      }
      return num_restored > 0;
    }
  } else {
    Log *log(GetLog(GDBRLog::Thread | GDBRLog::Packets));
    if (log) {
      if (log->GetVerbose()) {
        StreamString strm;
        process->DumpPluginHistory(strm);
        LLDB_LOGF(log,
                  "error: failed to get packet sequence mutex, not sending "
                  "write all registers:\n%s",
                  strm.GetData());
      } else
        LLDB_LOGF(log,
                  "error: failed to get packet sequence mutex, not sending "
                  "write all registers");
    }
  }
  return false;
}

bool GDBRemoteRegisterContext::RegisterWriteCausesReconfigure(
    const llvm::StringRef name) {
  ExecutionContext exe_ctx(CalculateThread());
  const Architecture *architecture =
      exe_ctx.GetProcessRef().GetTarget().GetArchitecturePlugin();
  return architecture && architecture->RegisterWriteCausesReconfigure(name);
}

bool GDBRemoteRegisterContext::ReconfigureRegisterLayout() {
  ExecutionContext exe_ctx(CalculateThread());
  const Architecture *architecture =
      exe_ctx.GetProcessRef().GetTarget().GetArchitecturePlugin();
  if (!architecture || !architecture->ReconfigureRegisterLayout(
                           m_register_layout_sp, m_reg_data, *this))
    return false;
  SetRegisterDescription(m_register_layout_sp->GetDescription());
  InvalidateAllRegisters();
  return true;
}
