//===-- ABISysV_loongarch.cpp----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABISysV_loongarch.h"

#include <array>
#include <limits>
#include <sstream>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/MathExtras.h"

#include "Utility/LoongArch_DWARF_Registers.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"

#define DEFINE_REG_NAME(reg_num) ConstString(#reg_num).GetCString()
#define DEFINE_REG_NAME_STR(reg_name) ConstString(reg_name).GetCString()

// The ABI is not a source of such information as size, offset, encoding, etc.
// of a register. Just provides correct dwarf and eh_frame numbers.

#define DEFINE_GENERIC_REGISTER_STUB(dwarf_num, generic_num)                   \
  {                                                                            \
      DEFINE_REG_NAME(dwarf_num),                                              \
      DEFINE_REG_NAME_STR(nullptr),                                            \
      0,                                                                       \
      0,                                                                       \
      eEncodingInvalid,                                                        \
      eFormatDefault,                                                          \
      {dwarf_num, dwarf_num, generic_num, LLDB_INVALID_REGNUM, dwarf_num},     \
      nullptr,                                                                 \
      nullptr,                                                                 \
      nullptr,                                                                 \
  }

#define DEFINE_REGISTER_STUB(dwarf_num)                                        \
  DEFINE_GENERIC_REGISTER_STUB(dwarf_num, LLDB_INVALID_REGNUM)

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(ABISysV_loongarch, ABILoongArch)

namespace {
namespace dwarf {
enum regnums {
  r0,
  r1,
  ra = r1,
  r2,
  r3,
  sp = r3,
  r4,
  r5,
  r6,
  r7,
  r8,
  r9,
  r10,
  r11,
  r12,
  r13,
  r14,
  r15,
  r16,
  r17,
  r18,
  r19,
  r20,
  r21,
  r22,
  fp = r22,
  r23,
  r24,
  r25,
  r26,
  r27,
  r28,
  r29,
  r30,
  r31,
  pc
};

static const std::array<RegisterInfo, 33> g_register_infos = {
    {DEFINE_REGISTER_STUB(r0),
     DEFINE_GENERIC_REGISTER_STUB(r1, LLDB_REGNUM_GENERIC_RA),
     DEFINE_REGISTER_STUB(r2),
     DEFINE_GENERIC_REGISTER_STUB(r3, LLDB_REGNUM_GENERIC_SP),
     DEFINE_GENERIC_REGISTER_STUB(r4, LLDB_REGNUM_GENERIC_ARG1),
     DEFINE_GENERIC_REGISTER_STUB(r5, LLDB_REGNUM_GENERIC_ARG2),
     DEFINE_GENERIC_REGISTER_STUB(r6, LLDB_REGNUM_GENERIC_ARG3),
     DEFINE_GENERIC_REGISTER_STUB(r7, LLDB_REGNUM_GENERIC_ARG4),
     DEFINE_GENERIC_REGISTER_STUB(r8, LLDB_REGNUM_GENERIC_ARG5),
     DEFINE_GENERIC_REGISTER_STUB(r9, LLDB_REGNUM_GENERIC_ARG6),
     DEFINE_GENERIC_REGISTER_STUB(r10, LLDB_REGNUM_GENERIC_ARG7),
     DEFINE_GENERIC_REGISTER_STUB(r11, LLDB_REGNUM_GENERIC_ARG8),
     DEFINE_REGISTER_STUB(r12),
     DEFINE_REGISTER_STUB(r13),
     DEFINE_REGISTER_STUB(r14),
     DEFINE_REGISTER_STUB(r15),
     DEFINE_REGISTER_STUB(r16),
     DEFINE_REGISTER_STUB(r17),
     DEFINE_REGISTER_STUB(r18),
     DEFINE_REGISTER_STUB(r19),
     DEFINE_REGISTER_STUB(r20),
     DEFINE_REGISTER_STUB(r21),
     DEFINE_GENERIC_REGISTER_STUB(r22, LLDB_REGNUM_GENERIC_FP),
     DEFINE_REGISTER_STUB(r23),
     DEFINE_REGISTER_STUB(r24),
     DEFINE_REGISTER_STUB(r25),
     DEFINE_REGISTER_STUB(r26),
     DEFINE_REGISTER_STUB(r27),
     DEFINE_REGISTER_STUB(r28),
     DEFINE_REGISTER_STUB(r29),
     DEFINE_REGISTER_STUB(r30),
     DEFINE_REGISTER_STUB(r31),
     DEFINE_GENERIC_REGISTER_STUB(pc, LLDB_REGNUM_GENERIC_PC)}};
} // namespace dwarf
} // namespace

// Number of argument registers (the base integer calling convention
// provides 8 argument registers, a0-a7)
static constexpr size_t g_regs_for_args_count = 8U;

const RegisterInfo *ABISysV_loongarch::GetRegisterInfoArray(uint32_t &count) {
  count = dwarf::g_register_infos.size();
  return dwarf::g_register_infos.data();
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------

ABISP
ABISysV_loongarch::CreateInstance(ProcessSP process_sp, const ArchSpec &arch) {
  llvm::Triple::ArchType machine = arch.GetTriple().getArch();

  if (llvm::Triple::loongarch32 != machine &&
      llvm::Triple::loongarch64 != machine)
    return ABISP();

  ABISysV_loongarch *abi =
      new ABISysV_loongarch(std::move(process_sp), MakeMCRegisterInfo(arch));
  if (abi)
    abi->SetIsLA64(llvm::Triple::loongarch64 == machine);
  return ABISP(abi);
}

static bool UpdateRegister(RegisterContext *reg_ctx,
                           const lldb::RegisterKind reg_kind,
                           const uint32_t reg_num, const addr_t value) {
  Log *log = GetLog(LLDBLog::Expressions);

  const RegisterInfo *reg_info = reg_ctx->GetRegisterInfo(reg_kind, reg_num);

  LLDB_LOG(log, "Writing {0}: 0x{1:x}", reg_info->name,
           static_cast<uint64_t>(value));
  if (!reg_ctx->WriteRegisterFromUnsigned(reg_info, value)) {
    LLDB_LOG(log, "Writing {0}: failed", reg_info->name);
    return false;
  }
  return true;
}

static void LogInitInfo(Log &log, const Thread &thread, addr_t sp,
                        addr_t func_addr, addr_t return_addr,
                        const llvm::ArrayRef<addr_t> args) {
  std::stringstream ss;
  ss << "ABISysV_loongarch::PrepareTrivialCall"
     << " (tid = 0x" << std::hex << thread.GetID() << ", sp = 0x" << sp
     << ", func_addr = 0x" << func_addr << ", return_addr = 0x" << return_addr;

  for (auto [idx, arg] : enumerate(args))
    ss << ", arg" << std::dec << idx << " = 0x" << std::hex << arg;
  ss << ")";
  log.PutString(ss.str());
}

bool ABISysV_loongarch::PrepareTrivialCall(Thread &thread, addr_t sp,
                                           addr_t func_addr, addr_t return_addr,
                                           llvm::ArrayRef<addr_t> args) const {
  Log *log = GetLog(LLDBLog::Expressions);
  if (log)
    LogInitInfo(*log, thread, sp, func_addr, return_addr, args);

  const auto reg_ctx_sp = thread.GetRegisterContext();
  if (!reg_ctx_sp) {
    LLDB_LOG(log, "Failed to get RegisterContext");
    return false;
  }

  if (args.size() > g_regs_for_args_count) {
    LLDB_LOG(log, "Function has {0} arguments, but only {1} are allowed!",
             args.size(), g_regs_for_args_count);
    return false;
  }

  // Write arguments to registers
  for (auto [idx, arg] : enumerate(args)) {
    const RegisterInfo *reg_info = reg_ctx_sp->GetRegisterInfo(
        eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1 + idx);
    LLDB_LOG(log, "About to write arg{0} ({1:x}) into {2}", idx, arg,
             reg_info->name);

    if (!reg_ctx_sp->WriteRegisterFromUnsigned(reg_info, arg)) {
      LLDB_LOG(log, "Failed to write arg{0} ({1:x}) into {2}", idx, arg,
               reg_info->name);
      return false;
    }
  }

  if (!UpdateRegister(reg_ctx_sp.get(), eRegisterKindGeneric,
                      LLDB_REGNUM_GENERIC_PC, func_addr))
    return false;
  if (!UpdateRegister(reg_ctx_sp.get(), eRegisterKindGeneric,
                      LLDB_REGNUM_GENERIC_SP, sp))
    return false;
  if (!UpdateRegister(reg_ctx_sp.get(), eRegisterKindGeneric,
                      LLDB_REGNUM_GENERIC_RA, return_addr))
    return false;

  LLDB_LOG(log, "ABISysV_loongarch::{0}() success", __FUNCTION__);
  return true;
}

bool ABISysV_loongarch::GetArgumentValues(Thread &thread,
                                          ValueList &values) const {
  // TODO: Implement
  return false;
}

Status ABISysV_loongarch::SetReturnValueObject(StackFrameSP &frame_sp,
                                               ValueObjectSP &new_value_sp) {
  Status result;
  if (!new_value_sp) {
    result = Status::FromErrorString("Empty value object for return value.");
    return result;
  }

  CompilerType compiler_type = new_value_sp->GetCompilerType();
  if (!compiler_type) {
    result = Status::FromErrorString("Null clang type for return value.");
    return result;
  }

  auto &reg_ctx = *frame_sp->GetThread()->GetRegisterContext();

  bool is_signed = false;
  if (!compiler_type.IsIntegerOrEnumerationType(is_signed) &&
      !compiler_type.IsPointerType()) {
    result = Status::FromErrorString(
        "We don't support returning other types at present");
    return result;
  }

  DataExtractor data;
  size_t num_bytes = new_value_sp->GetData(data, result);

  if (result.Fail()) {
    result = Status::FromErrorStringWithFormat(
        "Couldn't convert return value to raw data: %s", result.AsCString());
    return result;
  }

  size_t reg_size = m_is_la64 ? 8 : 4;
  // Currently, we only support sizeof(data) <= 2 * reg_size.
  // 1. If the (`size` <= reg_size), the `data` will be returned through `ARG1`.
  // 2. If the (`size` > reg_size && `size` <= 2 * reg_size), the `data` will be
  // returned through a pair of registers (ARG1 and ARG2), and the lower-ordered
  // bits in the `ARG1`.
  if (num_bytes > 2 * reg_size) {
    result = Status::FromErrorString(
        "We don't support returning large integer values at present.");
    return result;
  }

  offset_t offset = 0;
  uint64_t raw_value = data.GetMaxU64(&offset, num_bytes);
  // According to psABI, i32 (no matter signed or unsigned) should be
  // sign-extended in register.
  if (4 == num_bytes && m_is_la64)
    raw_value = llvm::SignExtend64<32>(raw_value);
  auto reg_info =
      reg_ctx.GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1);
  if (!reg_ctx.WriteRegisterFromUnsigned(reg_info, raw_value)) {
    result = Status::FromErrorStringWithFormat(
        "Couldn't write value to register %s", reg_info->name);
    return result;
  }

  if (num_bytes <= reg_size)
    return result; // Successfully written.

  // For loongarch32, get the upper 32 bits from raw_value and write them.
  // For loongarch64, get the next 64 bits from data and write them.
  if (4 == reg_size)
    raw_value >>= 32;
  else
    raw_value = data.GetMaxU64(&offset, num_bytes - reg_size);

  reg_info =
      reg_ctx.GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG2);
  if (!reg_ctx.WriteRegisterFromUnsigned(reg_info, raw_value))
    result = Status::FromErrorStringWithFormat(
        "Couldn't write value to register %s", reg_info->name);

  return result;
}

template <typename T>
static void SetInteger(Scalar &scalar, uint64_t raw_value, bool is_signed) {
  static_assert(std::is_unsigned<T>::value, "T must be an unsigned type.");
  raw_value &= std::numeric_limits<T>::max();
  if (is_signed)
    scalar = static_cast<typename std::make_signed<T>::type>(raw_value);
  else
    scalar = static_cast<T>(raw_value);
}

static bool SetSizedInteger(Scalar &scalar, uint64_t raw_value,
                            uint8_t size_in_bytes, bool is_signed) {
  switch (size_in_bytes) {
  default:
    return false;

  case sizeof(uint64_t):
    SetInteger<uint64_t>(scalar, raw_value, is_signed);
    break;

  case sizeof(uint32_t):
    SetInteger<uint32_t>(scalar, raw_value, is_signed);
    break;

  case sizeof(uint16_t):
    SetInteger<uint16_t>(scalar, raw_value, is_signed);
    break;

  case sizeof(uint8_t):
    SetInteger<uint8_t>(scalar, raw_value, is_signed);
    break;
  }

  return true;
}

static bool SetSizedFloat(Scalar &scalar, uint64_t raw_value,
                          uint8_t size_in_bytes) {
  switch (size_in_bytes) {
  default:
    return false;

  case sizeof(uint64_t):
    scalar = *reinterpret_cast<double *>(&raw_value);
    break;

  case sizeof(uint32_t):
    scalar = *reinterpret_cast<float *>(&raw_value);
    break;
  }

  return true;
}

static ValueObjectSP GetValObjFromIntRegs(Thread &thread,
                                          const RegisterContextSP &reg_ctx,
                                          llvm::Triple::ArchType machine,
                                          uint32_t type_flags,
                                          uint32_t byte_size) {
  Value value;
  ValueObjectSP return_valobj_sp;
  auto *reg_info_a0 =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1);
  auto *reg_info_a1 =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG2);
  uint64_t raw_value = 0;

  switch (byte_size) {
  case sizeof(uint32_t):
    // Read a0 to get the arg
    raw_value = reg_ctx->ReadRegisterAsUnsigned(reg_info_a0, 0) & UINT32_MAX;
    break;
  case sizeof(uint64_t):
    // Read a0 to get the arg on loongarch64, a0 and a1 on loongarch32
    if (llvm::Triple::loongarch32 == machine) {
      raw_value = reg_ctx->ReadRegisterAsUnsigned(reg_info_a0, 0) & UINT32_MAX;
      raw_value |=
          (reg_ctx->ReadRegisterAsUnsigned(reg_info_a1, 0) & UINT32_MAX) << 32U;
    } else {
      raw_value = reg_ctx->ReadRegisterAsUnsigned(reg_info_a0, 0);
    }
    break;
  case 16: {
    // Read a0 and a1 to get the arg on loongarch64, not supported on
    // loongarch32
    if (llvm::Triple::loongarch32 == machine)
      return return_valobj_sp;

    // Create the ValueObjectSP here and return
    std::unique_ptr<DataBufferHeap> heap_data_up(
        new DataBufferHeap(byte_size, 0));
    const ByteOrder byte_order = thread.GetProcess()->GetByteOrder();
    RegisterValue reg_value_a0, reg_value_a1;
    if (reg_ctx->ReadRegister(reg_info_a0, reg_value_a0) &&
        reg_ctx->ReadRegister(reg_info_a1, reg_value_a1)) {
      Status error;
      if (reg_value_a0.GetAsMemoryData(*reg_info_a0,
                                       heap_data_up->GetBytes() + 0, 8,
                                       byte_order, error) &&
          reg_value_a1.GetAsMemoryData(*reg_info_a1,
                                       heap_data_up->GetBytes() + 8, 8,
                                       byte_order, error)) {
        value.SetBytes(heap_data_up.release(), byte_size);
        return ValueObjectConstResult::Create(
            thread.GetStackFrameAtIndex(0).get(), value, ConstString(""));
      }
    }
    break;
  }
  default:
    return return_valobj_sp;
  }

  if (type_flags & eTypeIsInteger) {
    if (!SetSizedInteger(value.GetScalar(), raw_value, byte_size,
                         type_flags & eTypeIsSigned))
      return return_valobj_sp;
  } else if (type_flags & eTypeIsFloat) {
    if (!SetSizedFloat(value.GetScalar(), raw_value, byte_size))
      return return_valobj_sp;
  } else
    return return_valobj_sp;

  value.SetValueType(Value::ValueType::Scalar);
  return_valobj_sp = ValueObjectConstResult::Create(
      thread.GetStackFrameAtIndex(0).get(), value, ConstString(""));
  return return_valobj_sp;
}

static ValueObjectSP GetValObjFromFPRegs(Thread &thread,
                                         const RegisterContextSP &reg_ctx,
                                         llvm::Triple::ArchType machine,
                                         uint32_t type_flags,
                                         uint32_t byte_size) {
  auto *reg_info_fa0 = reg_ctx->GetRegisterInfoByName("f0");
  bool use_fp_regs = false;
  ValueObjectSP return_valobj_sp;

  if (byte_size <= 8)
    use_fp_regs = true;

  if (use_fp_regs) {
    uint64_t raw_value;
    Value value;
    raw_value = reg_ctx->ReadRegisterAsUnsigned(reg_info_fa0, 0);
    if (!SetSizedFloat(value.GetScalar(), raw_value, byte_size))
      return return_valobj_sp;
    value.SetValueType(Value::ValueType::Scalar);
    return ValueObjectConstResult::Create(thread.GetStackFrameAtIndex(0).get(),
                                          value, ConstString(""));
  }
  // we should never reach this, but if we do, use the integer registers
  return GetValObjFromIntRegs(thread, reg_ctx, machine, type_flags, byte_size);
}

ValueObjectSP ABISysV_loongarch::GetReturnValueObjectSimple(
    Thread &thread, CompilerType &compiler_type) const {
  ValueObjectSP return_valobj_sp;

  if (!compiler_type)
    return return_valobj_sp;

  auto reg_ctx = thread.GetRegisterContext();
  if (!reg_ctx)
    return return_valobj_sp;

  Value value;
  value.SetCompilerType(compiler_type);

  const uint32_t type_flags = compiler_type.GetTypeInfo();
  const size_t byte_size =
      llvm::expectedToOptional(compiler_type.GetByteSize(&thread)).value_or(0);
  const ArchSpec arch = thread.GetProcess()->GetTarget().GetArchitecture();
  const llvm::Triple::ArchType machine = arch.GetMachine();

  if (type_flags & eTypeIsInteger) {
    return_valobj_sp =
        GetValObjFromIntRegs(thread, reg_ctx, machine, type_flags, byte_size);
    return return_valobj_sp;
  }
  if (type_flags & eTypeIsPointer) {
    const auto *reg_info_a0 = reg_ctx->GetRegisterInfo(
        eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1);
    value.GetScalar() = reg_ctx->ReadRegisterAsUnsigned(reg_info_a0, 0);
    value.SetValueType(Value::ValueType::Scalar);
    return ValueObjectConstResult::Create(thread.GetStackFrameAtIndex(0).get(),
                                          value, ConstString(""));
  }
  if (type_flags & eTypeIsFloat) {
    uint32_t float_count = 0;
    bool is_complex = false;

    if (compiler_type.IsFloatingPointType(float_count, is_complex) &&
        float_count == 1 && !is_complex) {
      return_valobj_sp =
          GetValObjFromFPRegs(thread, reg_ctx, machine, type_flags, byte_size);
      return return_valobj_sp;
    }
  }
  return return_valobj_sp;
}

ValueObjectSP ABISysV_loongarch::GetReturnValueObjectImpl(
    Thread &thread, CompilerType &return_compiler_type) const {
  ValueObjectSP return_valobj_sp;

  if (!return_compiler_type)
    return return_valobj_sp;

  ExecutionContext exe_ctx(thread.shared_from_this());
  return GetReturnValueObjectSimple(thread, return_compiler_type);
}

UnwindPlanSP ABISysV_loongarch::CreateFunctionEntryUnwindPlan() {
  uint32_t pc_reg_num = loongarch_dwarf::dwarf_gpr_pc;
  uint32_t sp_reg_num = loongarch_dwarf::dwarf_gpr_sp;
  uint32_t ra_reg_num = loongarch_dwarf::dwarf_gpr_ra;

  UnwindPlan::Row row;

  // Define CFA as the stack pointer
  row.GetCFAValue().SetIsRegisterPlusOffset(sp_reg_num, 0);

  // Previous frame's pc is in ra
  row.SetRegisterLocationToRegister(pc_reg_num, ra_reg_num, true);

  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindDWARF);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("loongarch function-entry unwind plan");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  return plan_sp;
}

UnwindPlanSP ABISysV_loongarch::CreateDefaultUnwindPlan() {
  uint32_t pc_reg_num = LLDB_REGNUM_GENERIC_PC;
  uint32_t fp_reg_num = LLDB_REGNUM_GENERIC_FP;

  UnwindPlan::Row row;

  // Define the CFA as the current frame pointer value.
  row.GetCFAValue().SetIsRegisterPlusOffset(fp_reg_num, 0);

  int reg_size = 4;
  if (m_is_la64)
    reg_size = 8;

  // Assume the ra reg (return pc) and caller's frame pointer
  // have been spilled to stack already.
  row.SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, reg_size * -2, true);
  row.SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, reg_size * -1, true);

  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindGeneric);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("loongarch default unwind plan");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  plan_sp->SetUnwindPlanValidAtAllInstructions(eLazyBoolNo);
  return plan_sp;
}

bool ABISysV_loongarch::RegisterIsVolatile(const RegisterInfo *reg_info) {
  return !RegisterIsCalleeSaved(reg_info);
}

bool ABISysV_loongarch::RegisterIsCalleeSaved(const RegisterInfo *reg_info) {
  if (!reg_info)
    return false;

  const char *name = reg_info->name;
  ArchSpec arch = GetProcessSP()->GetTarget().GetArchitecture();
  uint32_t arch_flags = arch.GetFlags();
  // Floating point registers are only callee saved when using
  // F or D hardware floating point ABIs.
  bool is_hw_fp = (arch_flags & ArchSpec::eLoongArch_abi_mask) != 0;

  return llvm::StringSwitch<bool>(name)
      // integer ABI names
      .Cases({"ra", "sp", "fp"}, true)
      .Cases({"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"}, true)
      // integer hardware names
      .Cases({"r1", "r3", "r22"}, true)
      .Cases({"r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30", "31"},
             true)
      // floating point ABI names
      .Cases({"fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7"}, is_hw_fp)
      // floating point hardware names
      .Cases({"f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31"}, is_hw_fp)
      .Default(false);
}

void ABISysV_loongarch::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "System V ABI for LoongArch targets",
                                CreateInstance);
}

void ABISysV_loongarch::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

static uint32_t GetGenericNum(llvm::StringRef name) {
  return llvm::StringSwitch<uint32_t>(name)
      .Case("pc", LLDB_REGNUM_GENERIC_PC)
      .Cases("ra", "r1", LLDB_REGNUM_GENERIC_RA)
      .Cases("sp", "r3", LLDB_REGNUM_GENERIC_SP)
      .Cases("fp", "r22", LLDB_REGNUM_GENERIC_FP)
      .Cases("a0", "r4", LLDB_REGNUM_GENERIC_ARG1)
      .Cases("a1", "r5", LLDB_REGNUM_GENERIC_ARG2)
      .Cases("a2", "r6", LLDB_REGNUM_GENERIC_ARG3)
      .Cases("a3", "r7", LLDB_REGNUM_GENERIC_ARG4)
      .Cases("a4", "r8", LLDB_REGNUM_GENERIC_ARG5)
      .Cases("a5", "r9", LLDB_REGNUM_GENERIC_ARG6)
      .Cases("a6", "r10", LLDB_REGNUM_GENERIC_ARG7)
      .Cases("a7", "r11", LLDB_REGNUM_GENERIC_ARG8)
      .Default(LLDB_INVALID_REGNUM);
}

void ABISysV_loongarch::AugmentRegisterInfo(
    std::vector<lldb_private::DynamicRegisterInfo::Register> &regs) {
  lldb_private::RegInfoBasedABI::AugmentRegisterInfo(regs);

  static const llvm::StringMap<llvm::StringRef> isa_to_abi_alias_map = {
      {"r0", "zero"}, {"r1", "ra"},  {"r2", "tp"},  {"r3", "sp"},
      {"r4", "a0"},   {"r5", "a1"},  {"r6", "a2"},  {"r7", "a3"},
      {"r8", "a4"},   {"r9", "a5"},  {"r10", "a6"}, {"r11", "a7"},
      {"r12", "t0"},  {"r13", "t1"}, {"r14", "t2"}, {"r15", "t3"},
      {"r16", "t4"},  {"r17", "t5"}, {"r18", "t6"}, {"r19", "t7"},
      {"r20", "t8"},  {"r22", "fp"}, {"r23", "s0"}, {"r24", "s1"},
      {"r25", "s2"},  {"r26", "s3"}, {"r27", "s4"}, {"r28", "s5"},
      {"r29", "s6"},  {"r30", "s7"}, {"r31", "s8"}};

  for (auto it : llvm::enumerate(regs)) {
    llvm::StringRef reg_name = it.value().name.GetStringRef();

    // Set alt name for certain registers for convenience
    llvm::StringRef alias_name = isa_to_abi_alias_map.lookup(reg_name);
    if (!alias_name.empty())
      it.value().alt_name.SetString(alias_name);

    // Set generic regnum so lldb knows what the PC, etc is
    it.value().regnum_generic = GetGenericNum(reg_name);
  }
}
