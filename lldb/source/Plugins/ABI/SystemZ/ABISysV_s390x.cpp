//===-- ABISysV_s390x.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABISysV_s390x.h"

#include "lldb/Register/Arch/lldb-s390x-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-s390x-ehframe-register-enums.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/TargetParser/Triple.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include "lldb/ValueObject/ValueObjectMemory.h"
#include "lldb/ValueObject/ValueObjectRegister.h"
#include <optional>

namespace dwarf_regs = lldb_private::registers::s390x::dwarf;
namespace ehframe_regs = lldb_private::registers::s390x::ehframe;

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(ABISysV_s390x, ABISystemZ)

// RegisterKind: EHFrame, DWARF, Generic, Process Plugin, LLDB

#define DEFINE_REG(reg_id, name, size, alt, generic)                           \
  RegisterDescriptor(reg_id, #name, alt, size, eEncodingUint, eFormatHex,      \
                     {ehframe_regs::name, dwarf_regs::name, generic,           \
                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},               \
                     {}, {}, nullptr)

static const RegisterDescriptor g_register_infos[] = {
    DEFINE_REG(536870912, r0, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870913, r1, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870914, r2, 8, nullptr, LLDB_REGNUM_GENERIC_ARG1),
    DEFINE_REG(536870915, r3, 8, nullptr, LLDB_REGNUM_GENERIC_ARG2),
    DEFINE_REG(536870916, r4, 8, nullptr, LLDB_REGNUM_GENERIC_ARG3),
    DEFINE_REG(536870917, r5, 8, nullptr, LLDB_REGNUM_GENERIC_ARG4),
    DEFINE_REG(536870918, r6, 8, nullptr, LLDB_REGNUM_GENERIC_ARG5),
    DEFINE_REG(536870919, r7, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870920, r8, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870921, r9, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870922, r10, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870923, r11, 8, nullptr, LLDB_REGNUM_GENERIC_FP),
    DEFINE_REG(536870924, r12, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870925, r13, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870926, r14, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870927, r15, 8, "sp", LLDB_REGNUM_GENERIC_SP),
    DEFINE_REG(536870928, acr0, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870929, acr1, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870930, acr2, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870931, acr3, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870932, acr4, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870933, acr5, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870934, acr6, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870935, acr7, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870936, acr8, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870937, acr9, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870938, acr10, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870939, acr11, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870940, acr12, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870941, acr13, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870942, acr14, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870943, acr15, 4, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870944, pswm, 8, nullptr, LLDB_REGNUM_GENERIC_FLAGS),
    DEFINE_REG(536870945, pswa, 8, nullptr, LLDB_REGNUM_GENERIC_PC),
    DEFINE_REG(536870946, f0, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870947, f1, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870948, f2, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870949, f3, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870950, f4, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870951, f5, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870952, f6, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870953, f7, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870954, f8, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870955, f9, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870956, f10, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870957, f11, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870958, f12, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870959, f13, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870960, f14, 8, nullptr, LLDB_INVALID_REGNUM),
    DEFINE_REG(536870961, f15, 8, nullptr, LLDB_INVALID_REGNUM)};

static const uint32_t k_num_register_infos = std::size(g_register_infos);

const lldb_private::RegisterDescriptor *
ABISysV_s390x::GetRegisterInfoArray(uint32_t &count) {
  count = k_num_register_infos;
  return g_register_infos;
}

size_t ABISysV_s390x::GetRedZoneSize() const { return 0; }

// Static Functions

ABISP
ABISysV_s390x::CreateInstance(lldb::ProcessSP process_sp, const ArchSpec &arch) {
  if (arch.GetTriple().getArch() == llvm::Triple::systemz) {
    return ABISP(new ABISysV_s390x(std::move(process_sp), MakeMCRegisterInfo(arch)));
  }
  return ABISP();
}

bool ABISysV_s390x::PrepareTrivialCall(Thread &thread, addr_t sp,
                                       addr_t func_addr, addr_t return_addr,
                                       llvm::ArrayRef<addr_t> args) const {
  Log *log = GetLog(LLDBLog::Expressions);

  if (log) {
    StreamString s;
    s.Printf("ABISysV_s390x::PrepareTrivialCall (tid = 0x%" PRIx64
             ", sp = 0x%" PRIx64 ", func_addr = 0x%" PRIx64
             ", return_addr = 0x%" PRIx64,
             thread.GetID(), (uint64_t)sp, (uint64_t)func_addr,
             (uint64_t)return_addr);

    for (size_t i = 0; i < args.size(); ++i)
      s.Printf(", arg%" PRIu64 " = 0x%" PRIx64, static_cast<uint64_t>(i + 1),
               args[i]);
    s.PutCString(")");
    log->PutString(s.GetString());
  }

  ThreadRegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return false;

  const RegisterDescriptor *pc_reg_info =
      reg_ctx->GetRegisterByNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  const RegisterDescriptor *sp_reg_info =
      reg_ctx->GetRegisterByNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
  const RegisterDescriptor *ra_reg_info =
      reg_ctx->GetRegisterByName("r14", 0);
  ProcessSP process_sp(thread.GetProcess());

  // Allocate a new stack frame and space for stack arguments if necessary

  addr_t arg_pos = 0;
  if (args.size() > 5) {
    sp -= 8 * (args.size() - 5);
    arg_pos = sp;
  }

  sp -= 160;

  // Process arguments

  for (size_t i = 0; i < args.size(); ++i) {
    if (i < 5) {
      const RegisterDescriptor *reg_info = reg_ctx->GetRegisterByNumber(
          eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1 + i);
      LLDB_LOGF(log, "About to write arg%" PRIu64 " (0x%" PRIx64 ") into %s",
                static_cast<uint64_t>(i + 1), args[i], reg_info->name.c_str());
      if (llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
              (reg_info ? reg_info->id : LLDB_INVALID_REGNUM), args[i])))
        return false;
    } else {
      Status error;
      LLDB_LOGF(log, "About to write arg%" PRIu64 " (0x%" PRIx64 ") onto stack",
                static_cast<uint64_t>(i + 1), args[i]);
      if (!process_sp->WritePointerToMemory(arg_pos, args[i], error))
        return false;
      arg_pos += 8;
    }
  }

  // %r14 is set to the return address

  LLDB_LOGF(log, "Writing RA: 0x%" PRIx64, (uint64_t)return_addr);

  if (llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
          (ra_reg_info ? ra_reg_info->id : LLDB_INVALID_REGNUM), return_addr)))
    return false;

  // %r15 is set to the actual stack value.

  LLDB_LOGF(log, "Writing SP: 0x%" PRIx64, (uint64_t)sp);

  if (llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
          (sp_reg_info ? sp_reg_info->id : LLDB_INVALID_REGNUM), sp)))
    return false;

  // %pc is set to the address of the called function.

  LLDB_LOGF(log, "Writing PC: 0x%" PRIx64, (uint64_t)func_addr);

  if (llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
          (pc_reg_info ? pc_reg_info->id : LLDB_INVALID_REGNUM), func_addr)))
    return false;

  return true;
}

static bool ReadIntegerArgument(Scalar &scalar, unsigned int bit_width,
                                bool is_signed, Thread &thread,
                                uint32_t *argument_register_ids,
                                unsigned int &current_argument_register,
                                addr_t &current_stack_argument) {
  if (bit_width > 64)
    return false; // Scalar can't hold large integer arguments

  if (current_argument_register < 5) {
    scalar = thread.GetRegisterContext()->ReadRegisterAsUnsigned(
        argument_register_ids[current_argument_register], 0);
    current_argument_register++;
    if (is_signed)
      scalar.SignExtend(bit_width);
  } else {
    uint32_t byte_size = (bit_width + (8 - 1)) / 8;
    Status error;
    if (thread.GetProcess()->ReadScalarIntegerFromMemory(
            current_stack_argument + 8 - byte_size, byte_size, is_signed,
            scalar, error)) {
      current_stack_argument += 8;
      return true;
    }
    return false;
  }
  return true;
}

bool ABISysV_s390x::GetArgumentValues(Thread &thread, ValueList &values) const {
  unsigned int num_values = values.GetSize();
  unsigned int value_index;

  // Extract the register context so we can read arguments from registers

  ThreadRegisterContext *reg_ctx = thread.GetRegisterContext().get();

  if (!reg_ctx)
    return false;

  // Get the pointer to the first stack argument so we have a place to start
  // when reading data

  addr_t sp = reg_ctx->GetSP(0);

  if (!sp)
    return false;

  addr_t current_stack_argument = sp + 160;

  uint32_t argument_register_ids[5];

  argument_register_ids[0] =
      reg_ctx->GetRegisterByNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1)
          ->id;
  argument_register_ids[1] =
      reg_ctx->GetRegisterByNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG2)
          ->id;
  argument_register_ids[2] =
      reg_ctx->GetRegisterByNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG3)
          ->id;
  argument_register_ids[3] =
      reg_ctx->GetRegisterByNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG4)
          ->id;
  argument_register_ids[4] =
      reg_ctx->GetRegisterByNumber(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG5)
          ->id;

  unsigned int current_argument_register = 0;

  for (value_index = 0; value_index < num_values; ++value_index) {
    Value *value = values.GetValueAtIndex(value_index);

    if (!value)
      return false;

    // We currently only support extracting values with Clang QualTypes. Do we
    // care about others?
    CompilerType compiler_type = value->GetCompilerType();
    std::optional<uint64_t> bit_size =
        llvm::expectedToOptional(compiler_type.GetBitSize(&thread));
    if (!bit_size)
      return false;
    bool is_signed;

    if (compiler_type.IsIntegerOrEnumerationType(is_signed)) {
      ReadIntegerArgument(value->GetScalar(), *bit_size, is_signed, thread,
                          argument_register_ids, current_argument_register,
                          current_stack_argument);
    } else if (compiler_type.IsPointerType()) {
      ReadIntegerArgument(value->GetScalar(), *bit_size, false, thread,
                          argument_register_ids, current_argument_register,
                          current_stack_argument);
    }
  }

  return true;
}

Status ABISysV_s390x::SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                                           lldb::ValueObjectSP &new_value_sp) {
  Status error;
  if (!new_value_sp) {
    error = Status::FromErrorString("Empty value object for return value.");
    return error;
  }

  CompilerType compiler_type = new_value_sp->GetCompilerType();
  if (!compiler_type) {
    error = Status::FromErrorString("Null clang type for return value.");
    return error;
  }

  Thread *thread = frame_sp->GetThread().get();

  bool is_signed;

  ThreadRegisterContext *reg_ctx = thread->GetRegisterContext().get();

  bool set_it_simple = false;
  if (compiler_type.IsIntegerOrEnumerationType(is_signed) ||
      compiler_type.IsPointerType()) {
    const RegisterDescriptor *reg_info =
        reg_ctx->GetRegisterByName("r2", 0);

    DataExtractor data;
    Status data_error;
    size_t num_bytes = new_value_sp->GetData(data, data_error);
    if (data_error.Fail()) {
      error = Status::FromErrorStringWithFormat(
          "Couldn't convert return value to raw data: %s",
          data_error.AsCString());
      return error;
    }
    lldb::offset_t offset = 0;
    if (num_bytes <= 8) {
      uint64_t raw_value = data.GetMaxU64(&offset, num_bytes);

      if (!llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
              (reg_info ? reg_info->id : LLDB_INVALID_REGNUM), raw_value)))
        set_it_simple = true;
    } else {
      error = Status::FromErrorString(
          "We don't support returning longer than 64 bit "
          "integer values at present.");
    }
  } else if (compiler_type.IsRealFloatingPointType()) {
    std::optional<uint64_t> bit_width =
        llvm::expectedToOptional(compiler_type.GetBitSize(frame_sp.get()));
    if (!bit_width) {
      error = Status::FromErrorString("can't get type size");
      return error;
    }
    if (*bit_width <= 64) {
      const RegisterDescriptor *f0_info =
          reg_ctx->GetRegisterByName("f0", 0);
      RegisterValue f0_value;
      DataExtractor data;
      Status data_error;
      size_t num_bytes = new_value_sp->GetData(data, data_error);
      if (data_error.Fail()) {
        error = Status::FromErrorStringWithFormat(
            "Couldn't convert return value to raw data: %s",
            data_error.AsCString());
        return error;
      }

      unsigned char buffer[8];
      ByteOrder byte_order = data.GetByteOrder();

      data.CopyByteOrderedData(0, num_bytes, buffer, 8, byte_order);
      f0_value.SetBytes(buffer, 8, byte_order);
      llvm::consumeError(reg_ctx->WriteRegister(
          (f0_info ? f0_info->id : LLDB_INVALID_REGNUM), f0_value));
      set_it_simple = true;
    } else {
      // FIXME - don't know how to do long doubles yet.
      error = Status::FromErrorString(
          "We don't support returning float values > 64 bits at present");
    }
  }

  if (!set_it_simple) {
    // Okay we've got a structure or something that doesn't fit in a simple
    // register. We should figure out where it really goes, but we don't
    // support this yet.
    error = Status::FromErrorString(
        "We only support setting simple integer and float "
        "return types at present.");
  }

  return error;
}

ValueObjectSP ABISysV_s390x::GetReturnValueObjectSimple(
    Thread &thread, CompilerType &return_compiler_type) const {
  ValueObjectSP return_valobj_sp;
  Value value;

  if (!return_compiler_type)
    return return_valobj_sp;

  // value.SetContext (Value::eContextTypeClangType, return_value_type);
  value.SetCompilerType(return_compiler_type);

  ThreadRegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return return_valobj_sp;

  const uint32_t type_flags = return_compiler_type.GetTypeInfo();
  if (type_flags & eTypeIsScalar) {
    value.SetValueType(Value::ValueType::Scalar);

    bool success = false;
    if (type_flags & eTypeIsInteger) {
      // Extract the register context so we can read arguments from registers.
      std::optional<uint64_t> byte_size =
          llvm::expectedToOptional(return_compiler_type.GetByteSize(&thread));
      if (!byte_size)
        return return_valobj_sp;
      uint64_t raw_value = thread.GetRegisterContext()->ReadRegisterAsUnsigned(
          (reg_ctx->GetRegisterByName("r2", 0))->id, 0);
      const bool is_signed = (type_flags & eTypeIsSigned) != 0;
      switch (*byte_size) {
      default:
        break;

      case sizeof(uint64_t):
        if (is_signed)
          value.GetScalar() = (int64_t)(raw_value);
        else
          value.GetScalar() = (uint64_t)(raw_value);
        success = true;
        break;

      case sizeof(uint32_t):
        if (is_signed)
          value.GetScalar() = (int32_t)(raw_value & UINT32_MAX);
        else
          value.GetScalar() = (uint32_t)(raw_value & UINT32_MAX);
        success = true;
        break;

      case sizeof(uint16_t):
        if (is_signed)
          value.GetScalar() = (int16_t)(raw_value & UINT16_MAX);
        else
          value.GetScalar() = (uint16_t)(raw_value & UINT16_MAX);
        success = true;
        break;

      case sizeof(uint8_t):
        if (is_signed)
          value.GetScalar() = (int8_t)(raw_value & UINT8_MAX);
        else
          value.GetScalar() = (uint8_t)(raw_value & UINT8_MAX);
        success = true;
        break;
      }
    } else if (type_flags & eTypeIsFloat) {
      if (type_flags & eTypeIsComplex) {
        // Don't handle complex yet.
      } else {
        std::optional<uint64_t> byte_size =
            llvm::expectedToOptional(return_compiler_type.GetByteSize(&thread));
        if (byte_size && *byte_size <= sizeof(long double)) {
          const RegisterDescriptor *f0_info =
              reg_ctx->GetRegisterByName("f0", 0);
          RegisterValue f0_value;
          if (!llvm::errorToBool(
                  reg_ctx
                      ->ReadRegister(
                          (f0_info ? f0_info->id : LLDB_INVALID_REGNUM))
                      .moveInto(f0_value))) {
            DataExtractor data;
            if (f0_value.GetData(data)) {
              lldb::offset_t offset = 0;
              if (*byte_size == sizeof(float)) {
                value.GetScalar() = (float)data.GetFloat(&offset);
                success = true;
              } else if (*byte_size == sizeof(double)) {
                value.GetScalar() = (double)data.GetDouble(&offset);
                success = true;
              } else if (*byte_size == sizeof(long double)) {
                // Don't handle long double yet.
              }
            }
          }
        }
      }
    }

    if (success)
      return_valobj_sp = ValueObjectConstResult::Create(
          thread.GetStackFrameAtIndex(0).get(), value, ConstString(""));
  } else if (type_flags & eTypeIsPointer) {
    unsigned r2_id = reg_ctx->GetRegisterByName("r2", 0)->id;
    value.GetScalar() =
        (uint64_t)thread.GetRegisterContext()->ReadRegisterAsUnsigned(r2_id, 0);
    value.SetValueType(Value::ValueType::Scalar);
    return_valobj_sp = ValueObjectConstResult::Create(
        thread.GetStackFrameAtIndex(0).get(), value, ConstString(""));
  }

  return return_valobj_sp;
}

ValueObjectSP ABISysV_s390x::GetReturnValueObjectImpl(
    Thread &thread, CompilerType &return_compiler_type) const {
  ValueObjectSP return_valobj_sp;

  if (!return_compiler_type)
    return return_valobj_sp;

  ExecutionContext exe_ctx(thread.shared_from_this());
  return_valobj_sp = GetReturnValueObjectSimple(thread, return_compiler_type);
  if (return_valobj_sp)
    return return_valobj_sp;

  ThreadRegisterContextSP reg_ctx_sp = thread.GetRegisterContext();
  if (!reg_ctx_sp)
    return return_valobj_sp;

  if (return_compiler_type.IsAggregateType()) {
    // FIXME: This is just taking a guess, r2 may very well no longer hold the
    // return storage location.
    // If we are going to do this right, when we make a new frame we should
    // check to see if it uses a memory return, and if we are at the first
    // instruction and if so stash away the return location.  Then we would
    // only return the memory return value if we know it is valid.

    unsigned r2_id = reg_ctx_sp->GetRegisterByName("r2", 0)->id;
    lldb::addr_t storage_addr =
        (uint64_t)thread.GetRegisterContext()->ReadRegisterAsUnsigned(r2_id, 0);
    return_valobj_sp = ValueObjectMemory::Create(
        &thread, "", Address(storage_addr), return_compiler_type);
  }

  return return_valobj_sp;
}

UnwindPlanSP ABISysV_s390x::CreateFunctionEntryUnwindPlan() {
  UnwindPlan::Row row;

  // Our Call Frame Address is the stack pointer value + 160
  row.GetCFAValue().SetIsRegisterPlusOffset(dwarf_regs::r15, 160);

  // The previous PC is in r14
  row.SetRegisterLocationToRegister(dwarf_regs::pswa, dwarf_regs::r14, true);

  // All other registers are the same.
  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindDWARF);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("s390x at-func-entry default");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  return plan_sp;
}

UnwindPlanSP ABISysV_s390x::CreateDefaultUnwindPlan() {
  // There's really no default way to unwind on s390x. Trust the .eh_frame CFI,
  // which should always be good.
  return nullptr;
}

bool ABISysV_s390x::GetFallbackRegisterLocation(
    const RegisterDescriptor *reg_info,
    UnwindPlan::Row::AbstractRegisterLocation &unwind_regloc) {
  // If a volatile register is being requested, we don't want to forward the
  // next frame's register contents up the stack -- the register is not
  // retrievable at this frame.
  if (RegisterIsVolatile(reg_info)) {
    unwind_regloc.SetUndefined();
    return true;
  }

  return false;
}

bool ABISysV_s390x::RegisterIsVolatile(const RegisterDescriptor *reg_info) {
  return !RegisterIsCalleeSaved(reg_info);
}

bool ABISysV_s390x::RegisterIsCalleeSaved(const RegisterDescriptor *reg_info) {
  if (reg_info) {
    // Preserved registers are :
    //    r6-r13, r15
    //    f8-f15

    const char *name = reg_info->name.c_str();
    if (name[0] == 'r') {
      switch (name[1]) {
      case '6': // r6
      case '7': // r7
      case '8': // r8
      case '9': // r9
        return name[2] == '\0';

      case '1': // r10, r11, r12, r13, r15
        if ((name[2] >= '0' && name[2] <= '3') || name[2] == '5')
          return name[3] == '\0';
        break;

      default:
        break;
      }
    }
    if (name[0] == 'f') {
      switch (name[1]) {
      case '8': // r8
      case '9': // r9
        return name[2] == '\0';

      case '1': // r10, r11, r12, r13, r14, r15
        if (name[2] >= '0' && name[2] <= '5')
          return name[3] == '\0';
        break;

      default:
        break;
      }
    }

    // Accept shorter-variant versions
    if (name[0] == 's' && name[1] == 'p' && name[2] == '\0') // sp
      return true;
    if (name[0] == 'f' && name[1] == 'p' && name[2] == '\0') // fp
      return true;
    if (name[0] == 'p' && name[1] == 'c' && name[2] == '\0') // pc
      return true;
  }
  return false;
}

void ABISysV_s390x::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "System V ABI for s390x targets", CreateInstance);
}

void ABISysV_s390x::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
