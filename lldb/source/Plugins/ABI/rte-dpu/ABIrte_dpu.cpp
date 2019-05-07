//===-- ABIRrte_dpu.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ABIrte_dpu.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"

enum dwarf_regnums {
  dwarf_r0,
  dwarf_r1,
  dwarf_r2,
  dwarf_r3,
  dwarf_r4,
  dwarf_r5,
  dwarf_r6,
  dwarf_r7,
  dwarf_r8,
  dwarf_r9,
  dwarf_r10,
  dwarf_r11,
  dwarf_r12,
  dwarf_r13,
  dwarf_r14,
  dwarf_r15,
  dwarf_r16,
  dwarf_r17,
  dwarf_r18,
  dwarf_r19,
  dwarf_r20,
  dwarf_r21,
  dwarf_r22,
  dwarf_r23,
  dwarf_pc,
};

// todo double registers ? (safe registers ? constant registers ?)
static lldb_private::RegisterInfo g_register_infos[] = {
    {"r0",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r0, dwarf_r0, LLDB_REGNUM_GENERIC_ARG1, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r1",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r1, dwarf_r1, LLDB_REGNUM_GENERIC_ARG2, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r2",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r2, dwarf_r2, LLDB_REGNUM_GENERIC_ARG3, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r3",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r3, dwarf_r3, LLDB_REGNUM_GENERIC_ARG4, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r4",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r4, dwarf_r4, LLDB_REGNUM_GENERIC_ARG5, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r5",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r5, dwarf_r5, LLDB_REGNUM_GENERIC_ARG6, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r6",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r6, dwarf_r6, LLDB_REGNUM_GENERIC_ARG7, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r7",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r7, dwarf_r7, LLDB_REGNUM_GENERIC_ARG8, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r8",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r8, dwarf_r8, LLDB_REGNUM_GENERIC_ARG9, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r9",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r9, dwarf_r9, LLDB_REGNUM_GENERIC_ARG10, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r10",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r10, dwarf_r10, LLDB_REGNUM_GENERIC_ARG11, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r11",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r11, dwarf_r11, LLDB_REGNUM_GENERIC_ARG12, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r12",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r12, dwarf_r12, LLDB_REGNUM_GENERIC_ARG13, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r13",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r13, dwarf_r13, LLDB_REGNUM_GENERIC_ARG14, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r14",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r14, dwarf_r14, LLDB_REGNUM_GENERIC_ARG15, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r15",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r15, dwarf_r15, LLDB_REGNUM_GENERIC_ARG16, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r16",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r16, dwarf_r16, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r17",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r17, dwarf_r17, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r18",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r18, dwarf_r18, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r19",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r19, dwarf_r19, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r20",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r20, dwarf_r20, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r21",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r21, dwarf_r21, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r22",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r22, dwarf_r22, LLDB_REGNUM_GENERIC_SP, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"r23",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_r23, dwarf_r23, LLDB_REGNUM_GENERIC_RA, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
    {"pc",
     nullptr,
     4,
     0,
     lldb::eEncodingUint,
     lldb::eFormatHex,
     {dwarf_pc, dwarf_pc, LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM,
      LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0},
};

static const uint32_t k_num_register_infos =
    llvm::array_lengthof(g_register_infos);
static bool g_register_info_names_constified = false;

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------

size_t ABIrte_dpu::GetRedZoneSize() const { return 0; }

bool ABIrte_dpu::PrepareTrivialCall(lldb_private::Thread &thread,
                                    lldb::addr_t sp, lldb::addr_t func_addr,
                                    lldb::addr_t returnAddress,
                                    llvm::ArrayRef<lldb::addr_t> args) const {
  // todo handle 64-bit values ?
  lldb_private::RegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return false;

  const lldb_private::RegisterInfo *reg_info = nullptr;

  lldb_private::RegisterValue reg_value;

  const uint8_t reg_names[] = {
      LLDB_REGNUM_GENERIC_ARG1,  LLDB_REGNUM_GENERIC_ARG2,
      LLDB_REGNUM_GENERIC_ARG3,  LLDB_REGNUM_GENERIC_ARG4,
      LLDB_REGNUM_GENERIC_ARG5,  LLDB_REGNUM_GENERIC_ARG6,
      LLDB_REGNUM_GENERIC_ARG7,  LLDB_REGNUM_GENERIC_ARG8,
      LLDB_REGNUM_GENERIC_ARG9,  LLDB_REGNUM_GENERIC_ARG10,
      LLDB_REGNUM_GENERIC_ARG11, LLDB_REGNUM_GENERIC_ARG12,
      LLDB_REGNUM_GENERIC_ARG13, LLDB_REGNUM_GENERIC_ARG14,
      LLDB_REGNUM_GENERIC_ARG15, LLDB_REGNUM_GENERIC_ARG16,
  };

  llvm::ArrayRef<lldb::addr_t>::iterator ai = args.begin(), ae = args.end();

  // Write arguments to registers
  for (size_t i = 0; i < llvm::array_lengthof(reg_names); ++i) {
    if (ai == ae)
      break;

    reg_info =
        reg_ctx->GetRegisterInfo(lldb::eRegisterKindGeneric, reg_names[i]);

    if (!reg_ctx->WriteRegisterFromUnsigned(reg_info, args[i]))
      return false;

    ++ai;
  }

  // If we have more than 16 arguments --Spill onto the stack
  if (ai != ae) {
    // Number of arguments to go on stack
    size_t num_stack_regs = args.size();

    // Note: stack is growing up!
    lldb::addr_t arg_pos = sp;

    // just using arg1 to get the right size
    const lldb_private::RegisterInfo *reg_info = reg_ctx->GetRegisterInfo(
        lldb::eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1);

    // Allocate needed space for args on the stack
    sp += (num_stack_regs * reg_info->byte_size);

    // Keep the stack 8 byte aligned
    sp = (sp + (8ull - 1ull)) & ~(8ull - 1ull);

    for (; ai != ae; ++ai) {
      reg_value.SetUInt32(*ai);
      if (reg_ctx
              ->WriteRegisterValueToMemory(reg_info, arg_pos,
                                           reg_info->byte_size, reg_value)
              .Fail())
        return false;
      arg_pos += reg_info->byte_size;
    }
  }

  const lldb_private::RegisterInfo *pc_reg_info = reg_ctx->GetRegisterInfo(
      lldb::eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  const lldb_private::RegisterInfo *sp_reg_info = reg_ctx->GetRegisterInfo(
      lldb::eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
  const lldb_private::RegisterInfo *ra_reg_info = reg_ctx->GetRegisterInfo(
      lldb::eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA);

  // Set "sp" to the requested value
  if (!reg_ctx->WriteRegisterFromUnsigned(sp_reg_info, sp))
    return false;

  // Set "ra" to the return address
  if (!reg_ctx->WriteRegisterFromUnsigned(ra_reg_info, returnAddress))
    return false;

  // Set pc to the address of the called function.
  if (!reg_ctx->WriteRegisterFromUnsigned(pc_reg_info, func_addr))
    return false;

  return true;
}

bool ABIrte_dpu::GetArgumentValues(lldb_private::Thread &thread,
                                   lldb_private::ValueList &values) const {
  // todo
  return false;
}

lldb_private::Status
ABIrte_dpu::SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                                 lldb::ValueObjectSP &new_value_sp) {
  lldb_private::Status error;
  if (!new_value_sp) {
    error.SetErrorString("Empty value object for return value.");
    return error;
  }

  lldb_private::CompilerType compiler_type = new_value_sp->GetCompilerType();
  if (!compiler_type) {
    error.SetErrorString("Null clang type for return value.");
    return error;
  }

  lldb_private::Thread *thread = frame_sp->GetThread().get();

  bool is_signed;
  uint32_t count;
  bool is_complex;

  lldb_private::RegisterContext *reg_ctx = thread->GetRegisterContext().get();

  bool set_it_simple = false;
  if (compiler_type.IsIntegerOrEnumerationType(is_signed) ||
      compiler_type.IsPointerType()) {
    lldb_private::DataExtractor data;
    lldb_private::Status data_error;
    size_t num_bytes = new_value_sp->GetData(data, data_error);
    if (data_error.Fail()) {
      error.SetErrorStringWithFormat(
          "Couldn't convert return value to raw data: %s",
          data_error.AsCString());
      return error;
    }

    lldb::offset_t offset = 0;
    if (num_bytes <= 8) {
      const lldb_private::RegisterInfo *r21_info =
          reg_ctx->GetRegisterInfoByName("r21", 0);
      if (num_bytes <= 4) {
        uint32_t raw_value = data.GetMaxU32(&offset, num_bytes);

        if (reg_ctx->WriteRegisterFromUnsigned(r21_info, raw_value))
          set_it_simple = true;
      } else {
        uint32_t raw_value = data.GetMaxU32(&offset, 4);

        if (reg_ctx->WriteRegisterFromUnsigned(r21_info, raw_value)) {
          const lldb_private::RegisterInfo *r20_info =
              reg_ctx->GetRegisterInfoByName("r20", 0);
          uint32_t raw_value = data.GetMaxU32(&offset, num_bytes - offset);

          if (reg_ctx->WriteRegisterFromUnsigned(r20_info, raw_value))
            set_it_simple = true;
        }
      }
    } else {
      error.SetErrorString("We don't support returning longer than 64-bit "
                           "integer values at present.");
    }
  } else if (compiler_type.IsFloatingPointType(count, is_complex)) {
    if (is_complex)
      error.SetErrorString(
          "We don't support returning complex values at present.");
    else
      error.SetErrorString(
          "We don't support returning float values at present.");
  }

  if (!set_it_simple)
    error.SetErrorString(
        "We only support setting simple integer return types at present.");

  return error;
}

bool ABIrte_dpu::CreateFunctionEntryUnwindPlan(
    lldb_private::UnwindPlan &unwind_plan) {
  unwind_plan.Clear();
  unwind_plan.SetRegisterKind(lldb::eRegisterKindDWARF);

  uint32_t ra_reg_num = dwarf_r23;
  uint32_t sp_reg_num = dwarf_r22;
  uint32_t pc_reg_num = dwarf_pc;

  lldb_private::UnwindPlan::RowSP row(new lldb_private::UnwindPlan::Row);

  // Our Call Frame Address is the stack pointer value
  row->GetCFAValue().SetIsRegisterPlusOffset(sp_reg_num, 0);

  // The previous PC is in the RA
  row->SetRegisterLocationToRegister(pc_reg_num, ra_reg_num, true);
  unwind_plan.AppendRow(row);

  // All other registers are the same.

  unwind_plan.SetSourceName("dpu at-func-entry default");
  unwind_plan.SetSourcedFromCompiler(lldb_private::eLazyBoolNo);

  return true;
}

bool ABIrte_dpu::CreateDefaultUnwindPlan(
    lldb_private::UnwindPlan &unwind_plan) {
  unwind_plan.Clear();
  unwind_plan.SetRegisterKind(lldb::eRegisterKindDWARF);

  uint32_t ra_reg_num = dwarf_r23;
  uint32_t sp_reg_num = dwarf_r22;
  uint32_t pc_reg_num = dwarf_pc;

  lldb_private::UnwindPlan::RowSP row(new lldb_private::UnwindPlan::Row);

  row->GetCFAValue().SetIsRegisterPlusOffset(sp_reg_num, 0);

  row->SetRegisterLocationToRegister(pc_reg_num, ra_reg_num, true);

  unwind_plan.AppendRow(row);
  unwind_plan.SetSourceName("dpu default unwind plan");
  unwind_plan.SetSourcedFromCompiler(lldb_private::eLazyBoolNo);
  unwind_plan.SetUnwindPlanValidAtAllInstructions(lldb_private::eLazyBoolNo);
  return true;
}

bool ABIrte_dpu::RegisterIsVolatile(
    const lldb_private::RegisterInfo *reg_info) {
  return !RegisterIsCalleeSaved(reg_info);
}

bool ABIrte_dpu::CallFrameAddressIsValid(lldb::addr_t cfa) {
  // Make sure the stack call frame addresses are 8 byte aligned
  if (cfa & (8ull - 1ull))
    return false; // Not 8 byte aligned
  if (cfa == 0)
    return false; // Zero is not a valid stack address
  return true;
}

bool ABIrte_dpu::CodeAddressIsValid(lldb::addr_t pc) {
  return pc <= UINT32_MAX;
}

const lldb_private::RegisterInfo *
ABIrte_dpu::GetRegisterInfoArray(uint32_t &count) {
  // Make the C-string names and alt_names for the register infos into const
  // C-string values by having the ConstString unique the names in the global
  // constant C-string pool.
  if (!g_register_info_names_constified) {
    g_register_info_names_constified = true;
    for (uint32_t i = 0; i < k_num_register_infos; ++i) {
      if (g_register_infos[i].name)
        g_register_infos[i].name =
            lldb_private::ConstString(g_register_infos[i].name).GetCString();
      if (g_register_infos[i].alt_name)
        g_register_infos[i].alt_name =
            lldb_private::ConstString(g_register_infos[i].alt_name)
                .GetCString();
    }
  }
  count = k_num_register_infos;
  return g_register_infos;
}

lldb::ValueObjectSP ABIrte_dpu::GetReturnValueObjectImpl(
    lldb_private::Thread &thread,
    lldb_private::CompilerType &return_compiler_type) const {
  lldb::ValueObjectSP return_valobj_sp;
  lldb_private::Value value;

  if (!return_compiler_type)
    return return_valobj_sp;

  lldb_private::ExecutionContext exe_ctx(thread.shared_from_this());
  if (exe_ctx.GetTargetPtr() == nullptr || exe_ctx.GetProcessPtr() == nullptr)
    return return_valobj_sp;

  lldb::ByteOrder target_byte_order =
      exe_ctx.GetTargetPtr()->GetArchitecture().GetByteOrder();
  value.SetCompilerType(return_compiler_type);

  lldb_private::RegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return return_valobj_sp;

  bool is_signed = false;
  bool is_complex = false;
  uint32_t count = 0;

  // In DPU register "r21" holds the integer function return values
  const lldb_private::RegisterInfo *r21_reg_info =
      reg_ctx->GetRegisterInfoByName("r21", 0);
  size_t bit_width = return_compiler_type.GetBitSize(&thread);
  if (return_compiler_type.IsIntegerOrEnumerationType(is_signed)) {
    switch (bit_width) {
    default:
      return return_valobj_sp;
    case 64: {
      const lldb_private::RegisterInfo *r20_reg_info =
          reg_ctx->GetRegisterInfoByName("r20", 0);
      uint64_t raw_value;
      raw_value = reg_ctx->ReadRegisterAsUnsigned(r21_reg_info, 0) & UINT32_MAX;

      if (target_byte_order == lldb::eByteOrderLittle)
        raw_value = ((reg_ctx->ReadRegisterAsUnsigned(r20_reg_info, 0)) << 32) |
                    raw_value;
      else
        raw_value = (raw_value << 32) |
                    reg_ctx->ReadRegisterAsUnsigned(r20_reg_info, 0);

      if (is_signed)
        value.GetScalar() = (int64_t)raw_value;
      else
        value.GetScalar() = (uint64_t)raw_value;
    } break;
    case 32:
      if (is_signed)
        value.GetScalar() = (int32_t)(
            reg_ctx->ReadRegisterAsUnsigned(r21_reg_info, 0) & UINT32_MAX);
      else
        value.GetScalar() = (uint32_t)(
            reg_ctx->ReadRegisterAsUnsigned(r21_reg_info, 0) & UINT32_MAX);
      break;
    case 16:
      if (is_signed)
        value.GetScalar() = (int16_t)(
            reg_ctx->ReadRegisterAsUnsigned(r21_reg_info, 0) & UINT16_MAX);
      else
        value.GetScalar() = (uint16_t)(
            reg_ctx->ReadRegisterAsUnsigned(r21_reg_info, 0) & UINT16_MAX);
      break;
    case 8:
      if (is_signed)
        value.GetScalar() = (int8_t)(
            reg_ctx->ReadRegisterAsUnsigned(r21_reg_info, 0) & UINT8_MAX);
      else
        value.GetScalar() = (uint8_t)(
            reg_ctx->ReadRegisterAsUnsigned(r21_reg_info, 0) & UINT8_MAX);
      break;
    }
  } else if (return_compiler_type.IsPointerType()) {
    uint32_t ptr =
        thread.GetRegisterContext()->ReadRegisterAsUnsigned(r21_reg_info, 0) &
        UINT32_MAX;
    value.GetScalar() = ptr;
  } else if (return_compiler_type.IsAggregateType()) {
    // todo not handled yet
    return return_valobj_sp;
  } else if (return_compiler_type.IsFloatingPointType(count, is_complex)) {
    uint64_t raw_value = reg_ctx->ReadRegisterAsUnsigned(r21_reg_info, 0);
    if (count != 1 && is_complex)
      return return_valobj_sp;
    switch (bit_width) {
    default:
      return return_valobj_sp;
    case 32:
      static_assert(sizeof(float) == sizeof(uint32_t), "");
      value.GetScalar() = *((float *)(&raw_value));
      break;
    case 64:
      static_assert(sizeof(double) == sizeof(uint64_t), "");
      const lldb_private::RegisterInfo *r20_reg_info =
          reg_ctx->GetRegisterInfoByName("r20", 0);
      if (target_byte_order == lldb::eByteOrderLittle)
        raw_value = ((reg_ctx->ReadRegisterAsUnsigned(r20_reg_info, 0)) << 32) |
                    raw_value;
      else
        raw_value = (raw_value << 32) |
                    reg_ctx->ReadRegisterAsUnsigned(r20_reg_info, 0);
      value.GetScalar() = *((double *)(&raw_value));
      break;
    }
  } else {
    // todo not handled yet
    return return_valobj_sp;
  }

  // If we get here, we have a valid Value, so make our ValueObject out of it:
  return_valobj_sp = lldb_private::ValueObjectConstResult::Create(
      thread.GetStackFrameAtIndex(0).get(), value,
      lldb_private::ConstString(""));
  return return_valobj_sp;
}

bool ABIrte_dpu::RegisterIsCalleeSaved(
    const lldb_private::RegisterInfo *reg_info) {
  if (!reg_info)
    return false;
  assert(reg_info->name != nullptr && "unnamed register?");
  std::string Name = std::string(reg_info->name);
  bool IsCalleeSaved = llvm::StringSwitch<bool>(Name)
                           .Cases("r0", "r1", "r2", "r3", "r4", "r5", "r6",
                                  "r7", "r8", "r9", true)
                           .Cases("r10", "r11", "r12", "r13", "r14", "r15",
                                  "r16", "r22", "r23", true)
                           .Default(false);
  return IsCalleeSaved;
}

lldb::ABISP ABIrte_dpu::CreateInstance(lldb::ProcessSP process_sp,
                                       const lldb_private::ArchSpec &arch) {
  static lldb::ABISP g_abi_sp;
  const llvm::Triple::ArchType arch_type = arch.GetTriple().getArch();
  if ((arch_type == llvm::Triple::dpu)) {
    if (!g_abi_sp)
      g_abi_sp.reset(new ABIrte_dpu(process_sp));
    return g_abi_sp;
  }
  return lldb::ABISP();
}

void ABIrte_dpu::Initialize() {
  lldb_private::PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "RTE ABI for dpu targets", CreateInstance);
}

void ABIrte_dpu::Terminate() {
  lldb_private::PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString ABIrte_dpu::GetPluginNameStatic() {
  static lldb_private::ConstString g_name("rte-dpu");
  return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------

lldb_private::ConstString ABIrte_dpu::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ABIrte_dpu::GetPluginVersion() { return 1; }
