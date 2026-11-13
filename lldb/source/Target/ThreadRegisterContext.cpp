//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Scalar.h"

using namespace lldb;
using namespace lldb_private;

ThreadRegisterContext::ThreadRegisterContext(Thread &thread,
                                             uint32_t concrete_frame_idx)
    : RegisterContext(thread.shared_from_this()), m_thread(thread),
      m_concrete_frame_idx(concrete_frame_idx),
      m_stop_id(thread.GetProcess()->GetStopID()) {}

ThreadRegisterContext::ThreadRegisterContext(Thread &thread, uint32_t frame,
                                             RegisterLayout layout)
    : ThreadRegisterContext(thread, frame) {
  SetRegisterLayout(std::move(layout));
}

ThreadRegisterContext::ThreadRegisterContext(
    Thread &thread, uint32_t frame,
    std::shared_ptr<const RegisterDescription> description)
    : ThreadRegisterContext(thread, frame) {
  SetRegisterDescription(std::move(description));
}

void ThreadRegisterContext::SetRegisterDescription(
    std::shared_ptr<const RegisterDescription> description) {
  m_description = std::move(description);
  m_layout.reset();
}

void ThreadRegisterContext::SetRegisterLayout(RegisterLayout layout) {
  m_description = layout.GetDescription();
  m_layout = std::make_shared<RegisterLayout>(std::move(layout));
}

std::shared_ptr<const RegisterDescription>
ThreadRegisterContext::GetDescription() const {
  return m_description;
}

const RegisterLayout &ThreadRegisterContext::GetLayout() const {
  assert(m_layout && "context has no saved-register layout");
  return *m_layout;
}

const ArchSpec &ThreadRegisterContext::GetTargetArchitecture() const {
  return m_thread.GetProcess()->GetTarget().GetArchitecture();
}

size_t ThreadRegisterContext::GetBankSize(RegisterBankID id) const {
  const auto *bank = m_layout ? m_layout->FindBank(id) : nullptr;
  return bank ? bank->byte_size : 0;
}

llvm::Expected<RegisterValue> ThreadRegisterContext::ReadBankRegister(
    RegisterID id, llvm::ArrayRef<RegisterData> candidates) const {
  std::vector<RegisterData> banks;
  for (const auto &bank : candidates)
    if (GetLayout().FindBank(bank.bank))
      banks.push_back(bank);
  return GetLayout().Read(id, banks);
}

ThreadRegisterContext::~ThreadRegisterContext() = default;

void ThreadRegisterContext::InvalidateIfNeeded(bool force) {
  ProcessSP process_sp(m_thread.GetProcess());
  bool invalidate = force;
  uint32_t process_stop_id = UINT32_MAX;

  if (process_sp)
    process_stop_id = process_sp->GetStopID();
  else
    invalidate = true;

  if (!invalidate)
    invalidate = process_stop_id != GetStopID();

  if (invalidate) {
    InvalidateAllRegisters();
    SetStopID(process_stop_id);
  }
}

uint64_t ThreadRegisterContext::GetPC(uint64_t fail_value) {
  uint32_t reg =
      GetRegisterIDForNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  uint64_t pc = ReadRegisterAsUnsigned(reg, fail_value);

  if (pc != fail_value) {
    if (TargetSP target_sp = m_thread.CalculateTarget())
      pc = target_sp->GetOpcodeLoadAddress(pc, AddressClass::eCode);
  }

  return pc;
}

uint64_t ThreadRegisterContext::GetThreadPointer(uint64_t fail_value) {
  uint32_t reg =
      GetRegisterIDForNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_TP);
  return ReadRegisterAsUnsigned(reg, fail_value);
}

bool ThreadRegisterContext::SetPC(uint64_t pc) {
  uint32_t reg =
      GetRegisterIDForNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  bool success = !llvm::errorToBool(WriteRegisterFromUnsigned(reg, pc));
  if (success) {
    StackFrameSP frame_sp(
        m_thread.GetFrameWithConcreteFrameIndex(m_concrete_frame_idx));
    if (frame_sp)
      frame_sp->ChangePC(pc);
    else
      m_thread.ClearStackFrames();
  }
  return success;
}

bool ThreadRegisterContext::GetPCForSymbolication(Address &address) {
  addr_t pc = GetPC(LLDB_INVALID_ADDRESS);
  if (pc == LLDB_INVALID_ADDRESS)
    return false;
  TargetSP target_sp = m_thread.CalculateTarget();
  if (!target_sp.get())
    return false;

  if (!BehavesLikeZerothFrame() && pc != 0)
    pc--;
  address.SetLoadAddress(pc, target_sp.get());
  return true;
}

bool ThreadRegisterContext::SetPC(Address addr) {
  TargetSP target_sp = m_thread.CalculateTarget();
  Target *target = target_sp.get();

  lldb::addr_t callAddr = addr.GetCallableLoadAddress(target);
  if (callAddr == LLDB_INVALID_ADDRESS)
    return false;

  return SetPC(callAddr);
}

uint64_t ThreadRegisterContext::GetSP(uint64_t fail_value) {
  uint32_t reg =
      GetRegisterIDForNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
  return ReadRegisterAsUnsigned(reg, fail_value);
}

bool ThreadRegisterContext::SetSP(uint64_t sp) {
  uint32_t reg =
      GetRegisterIDForNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
  return !llvm::errorToBool(WriteRegisterFromUnsigned(reg, sp));
}

uint64_t ThreadRegisterContext::GetFP(uint64_t fail_value) {
  uint32_t reg =
      GetRegisterIDForNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP);
  return ReadRegisterAsUnsigned(reg, fail_value);
}

bool ThreadRegisterContext::SetFP(uint64_t fp) {
  uint32_t reg =
      GetRegisterIDForNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP);
  return !llvm::errorToBool(WriteRegisterFromUnsigned(reg, fp));
}

uint64_t ThreadRegisterContext::GetReturnAddress(uint64_t fail_value) {
  uint32_t reg =
      GetRegisterIDForNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA);
  return ReadRegisterAsUnsigned(reg, fail_value);
}

uint64_t ThreadRegisterContext::GetFlags(uint64_t fail_value) {
  uint32_t reg =
      GetRegisterIDForNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS);
  return ReadRegisterAsUnsigned(reg, fail_value);
}

bool ThreadRegisterContext::CopyFromRegisterContext(
    lldb::ThreadRegisterContextSP context) {
  uint32_t num_register_sets = context->GetRegisterGroupCount();
  // We don't know that two threads have the same register context, so require
  // the threads to be the same.
  if (context->GetThreadID() != GetThreadID())
    return false;

  if (num_register_sets != GetRegisterGroupCount())
    return false;

  ThreadRegisterContextSP frame_zero_context = m_thread.GetRegisterContext();

  for (uint32_t set_idx = 0; set_idx < num_register_sets; ++set_idx) {
    const RegisterGroup *const reg_set = GetRegisterGroup(set_idx);

    const uint32_t num_registers = reg_set->registers.size();
    for (uint32_t reg_idx = 0; reg_idx < num_registers; ++reg_idx) {
      const uint32_t reg = reg_set->registers[reg_idx];
      const RegisterDescriptor *reg_info = GetDescription()->Find(reg);
      if (!reg_info || !reg_info->value_registers.empty())
        continue;
      RegisterValue reg_value;

      // If we can reconstruct the register from the frame we are copying from,
      // then do so, otherwise use the value from frame 0.
      auto read = [&](ThreadRegisterContext &source) -> bool {
        const auto *source_reg = source.GetRegisterByName(reg_info->name);
        return source_reg &&
               !llvm::errorToBool(
                   source.ReadRegister(source_reg->id).moveInto(reg_value));
      };
      if (read(*context) || read(*frame_zero_context))
        llvm::consumeError(WriteRegister(reg_info->id, reg_value));
    }
  }
  return true;
}

lldb::tid_t ThreadRegisterContext::GetThreadID() const { return m_thread.GetID(); }

uint32_t ThreadRegisterContext::NumSupportedHardwareBreakpoints() { return 0; }

uint32_t ThreadRegisterContext::SetHardwareBreakpoint(lldb::addr_t addr,
                                                size_t size) {
  return LLDB_INVALID_INDEX32;
}

// Used when parsing DWARF and EH frame information and any other object file
// sections that contain register numbers in them.

bool ThreadRegisterContext::ClearHardwareBreakpoint(uint32_t hw_idx) {
  return false;
}

uint32_t ThreadRegisterContext::NumSupportedHardwareWatchpoints() { return 0; }

uint32_t ThreadRegisterContext::SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                                                bool read, bool write) {
  return LLDB_INVALID_INDEX32;
}

bool ThreadRegisterContext::ClearHardwareWatchpoint(uint32_t hw_index) {
  return false;
}

bool ThreadRegisterContext::HardwareSingleStep(bool enable) { return false; }

Status ThreadRegisterContext::ReadRegisterValueFromMemory(
    const RegisterDescriptor *reg_info, lldb::addr_t src_addr, uint32_t src_len,
    RegisterValue &reg_value) {
  Status error;
  if (!reg_info)
    return Status::FromErrorString("invalid register info argument.");

  // Moving from addr into a register
  //
  // Case 1: src_len == dst_len
  //
  //   |AABBCCDD| Address contents
  //   |AABBCCDD| Register contents
  //
  // Case 2: src_len > dst_len
  //
  //   Status!  (The register should always be big enough to hold the data)
  //
  // Case 3: src_len < dst_len
  //
  //   |AABB| Address contents
  //   |AABB0000| Register contents [on little-endian hardware]
  //   |0000AABB| Register contents [on big-endian hardware]
  const uint32_t dst_len = reg_info->byte_size;

  if (src_len > dst_len) {
    return Status::FromErrorStringWithFormat(
        "%u bytes is too big to store in register %s (%u bytes)", src_len,
        reg_info->name.c_str(), dst_len);
    return error;
  }

  ProcessSP process_sp(m_thread.GetProcess());
  if (process_sp) {
    RegisterValue::BytesContainer src(src_len);

    // Read the memory
    const uint32_t bytes_read =
        process_sp->ReadMemory(src_addr, src.data(), src_len, error);

    // Make sure the memory read succeeded...
    if (bytes_read != src_len) {
      if (error.Success()) {
        // This might happen if we read _some_ bytes but not all
        return Status::FromErrorStringWithFormat("read %u of %u bytes",
                                                 bytes_read, src_len);
      }
      return error;
    }

    // We now have a memory buffer that contains the part or all of the
    // register value. Set the register value using this memory data.
    // TODO: we might need to add a parameter to this function in case the byte
    // order of the memory data doesn't match the process. For now we are
    // assuming they are the same.
    reg_value.SetFromMemoryData(reg_info->encoding, reg_info->byte_size,
                                src.data(), src_len, process_sp->GetByteOrder(),
                                error, reg_info->name);
  } else
    return Status::FromErrorString("invalid process");

  return error;
}

Status ThreadRegisterContext::WriteRegisterValueToMemory(
    const RegisterDescriptor *reg_info, lldb::addr_t dst_addr, uint32_t dst_len,
    const RegisterValue &reg_value) {
  Status error;
  ProcessSP process_sp(m_thread.GetProcess());

  if (!process_sp) {
    return Status::FromErrorString("invalid process");
    return error;
  }

  if (reg_info == nullptr) {
    return Status::FromErrorString("Invalid register info argument.");
    return error;
  }

  // TODO: we might need to add a parameter to this function in case the byte
  // order of the memory data doesn't match the process. For now we are
  // assuming they are the same.
  RegisterValue::BytesContainer dst(dst_len);
  const uint32_t bytes_copied = reg_value.GetAsMemoryData(
      reg_info->byte_size, dst.data(), dst_len, process_sp->GetByteOrder(),
      error, reg_info->name);

  if (error.Success()) {
    if (bytes_copied == 0) {
      return Status::FromErrorString("byte copy failed.");
    } else {
      const uint32_t bytes_written =
          process_sp->WriteMemory(dst_addr, dst.data(), bytes_copied, error);
      if (bytes_written != bytes_copied) {
        if (error.Success()) {
          // This might happen if we read _some_ bytes but not all
          return Status::FromErrorStringWithFormat("only wrote %u of %u bytes",
                                                   bytes_written, bytes_copied);
        }
      }
    }
  }

  return error;
}

lldb::ByteOrder ThreadRegisterContext::GetByteOrder() {
  // Get the target process whose privileged thread was used for the register
  // read.
  lldb::ByteOrder byte_order = lldb::eByteOrderInvalid;
  lldb_private::Process *process = CalculateProcess().get();

  if (process)
    byte_order = process->GetByteOrder();
  return byte_order;
}

llvm::Expected<RegisterCheckpoint> ThreadRegisterContext::SaveCheckpoint() {
  WritableDataBufferSP data;
  if (!ReadAllRegisterValues(data))
    return llvm::createStringError("could not capture register state");
  auto description = GetDescription();
  return CreateCheckpoint(
      [source =
           std::static_pointer_cast<ThreadRegisterContext>(shared_from_this()),
       data = std::move(data), description]() -> llvm::Error {
        auto &context = *source;
        if (context.GetDescription() != description)
          return llvm::createStringError(
              "register configuration changed since checkpoint");
        if (!context.WriteAllRegisterValues(data))
          return llvm::createStringError("could not restore register state");
        return llvm::Error::success();
      });
}

TargetSP ThreadRegisterContext::CalculateTarget() {
  return m_thread.CalculateTarget();
}

ProcessSP ThreadRegisterContext::CalculateProcess() {
  return m_thread.CalculateProcess();
}

ThreadSP ThreadRegisterContext::CalculateThread() {
  return m_thread.shared_from_this();
}

StackFrameSP ThreadRegisterContext::CalculateStackFrame() {
  // Register contexts might belong to many frames if we have inlined functions
  // inside a frame since all inlined functions share the same registers, so we
  // can't definitively say which frame we come from...
  return StackFrameSP();
}

void ThreadRegisterContext::CalculateExecutionContext(ExecutionContext &exe_ctx) {
  m_thread.CalculateExecutionContext(exe_ctx);
}
