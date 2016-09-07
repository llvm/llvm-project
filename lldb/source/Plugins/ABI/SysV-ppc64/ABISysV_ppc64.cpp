//===-- ABISysV_ppc64.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ABISysV_ppc64.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"

// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

enum dwarf_regnums {
  dwarf_r0 = 0,
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
  dwarf_r24,
  dwarf_r25,
  dwarf_r26,
  dwarf_r27,
  dwarf_r28,
  dwarf_r29,
  dwarf_r30,
  dwarf_r31,
  dwarf_f0,
  dwarf_f1,
  dwarf_f2,
  dwarf_f3,
  dwarf_f4,
  dwarf_f5,
  dwarf_f6,
  dwarf_f7,
  dwarf_f8,
  dwarf_f9,
  dwarf_f10,
  dwarf_f11,
  dwarf_f12,
  dwarf_f13,
  dwarf_f14,
  dwarf_f15,
  dwarf_f16,
  dwarf_f17,
  dwarf_f18,
  dwarf_f19,
  dwarf_f20,
  dwarf_f21,
  dwarf_f22,
  dwarf_f23,
  dwarf_f24,
  dwarf_f25,
  dwarf_f26,
  dwarf_f27,
  dwarf_f28,
  dwarf_f29,
  dwarf_f30,
  dwarf_f31,
  dwarf_cr,
  dwarf_fpscr,
  dwarf_xer = 101,
  dwarf_lr = 108,
  dwarf_ctr,
  dwarf_pc,
  dwarf_cfa,
};

// Note that the size and offset will be updated by platform-specific classes.
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)                       \
  {                                                                            \
    #reg, alt, 8, 0, eEncodingUint, eFormatHex, {kind1, kind2, kind3, kind4 }, \
                                                 nullptr, nullptr, nullptr, 0  \
  }

static const RegisterInfo g_register_infos[] = {
    // General purpose registers.            eh_frame,                  DWARF,
    // Generic,     Process Plugin
    DEFINE_GPR(r0, nullptr, dwarf_r0, dwarf_r0, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r1, "sp", dwarf_r1, dwarf_r1, LLDB_REGNUM_GENERIC_SP,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r2, nullptr, dwarf_r2, dwarf_r2, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r3, "arg1", dwarf_r3, dwarf_r3, LLDB_REGNUM_GENERIC_ARG1,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r4, "arg2", dwarf_r4, dwarf_r4, LLDB_REGNUM_GENERIC_ARG2,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r5, "arg3", dwarf_r5, dwarf_r5, LLDB_REGNUM_GENERIC_ARG3,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r6, "arg4", dwarf_r6, dwarf_r6, LLDB_REGNUM_GENERIC_ARG4,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r7, "arg5", dwarf_r7, dwarf_r7, LLDB_REGNUM_GENERIC_ARG5,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r8, "arg6", dwarf_r8, dwarf_r8, LLDB_REGNUM_GENERIC_ARG6,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r9, "arg7", dwarf_r9, dwarf_r9, LLDB_REGNUM_GENERIC_ARG7,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r10, "arg8", dwarf_r10, dwarf_r10, LLDB_REGNUM_GENERIC_ARG8,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r11, nullptr, dwarf_r11, dwarf_r11, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r12, nullptr, dwarf_r12, dwarf_r12, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r13, nullptr, dwarf_r13, dwarf_r13, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r14, nullptr, dwarf_r14, dwarf_r14, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r15, nullptr, dwarf_r15, dwarf_r15, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r16, nullptr, dwarf_r16, dwarf_r16, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r17, nullptr, dwarf_r17, dwarf_r17, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r18, nullptr, dwarf_r18, dwarf_r18, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r19, nullptr, dwarf_r19, dwarf_r19, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r20, nullptr, dwarf_r20, dwarf_r20, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r21, nullptr, dwarf_r21, dwarf_r21, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r22, nullptr, dwarf_r22, dwarf_r22, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r23, nullptr, dwarf_r23, dwarf_r23, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r24, nullptr, dwarf_r24, dwarf_r24, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r25, nullptr, dwarf_r25, dwarf_r25, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r26, nullptr, dwarf_r26, dwarf_r26, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r27, nullptr, dwarf_r27, dwarf_r27, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r28, nullptr, dwarf_r28, dwarf_r28, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r29, nullptr, dwarf_r29, dwarf_r29, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r30, nullptr, dwarf_r30, dwarf_r30, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(r31, nullptr, dwarf_r31, dwarf_r31, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(lr, "lr", dwarf_lr, dwarf_lr, LLDB_REGNUM_GENERIC_RA,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(cr, "cr", dwarf_cr, dwarf_cr, LLDB_REGNUM_GENERIC_FLAGS,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(xer, "xer", dwarf_xer, dwarf_xer, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(ctr, "ctr", dwarf_ctr, dwarf_ctr, LLDB_INVALID_REGNUM,
               LLDB_INVALID_REGNUM),
    DEFINE_GPR(pc, "pc", dwarf_pc, dwarf_pc, LLDB_REGNUM_GENERIC_PC,
               LLDB_INVALID_REGNUM),
    {nullptr,
     nullptr,
     8,
     0,
     eEncodingUint,
     eFormatHex,
     {dwarf_cfa, dwarf_cfa, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
     nullptr,
     nullptr,
     nullptr,
     0}};

static const uint32_t k_num_register_infos =
    llvm::array_lengthof(g_register_infos);

const lldb_private::RegisterInfo *
ABISysV_ppc64::GetRegisterInfoArray(uint32_t &count) {
  count = k_num_register_infos;
  return g_register_infos;
}

size_t ABISysV_ppc64::GetRedZoneSize() const { return 224; }

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------

ABISP
ABISysV_ppc64::CreateInstance(const ArchSpec &arch) {
  static ABISP g_abi_sp;
  if (arch.GetTriple().getArch() == llvm::Triple::ppc64) {
    if (!g_abi_sp)
      g_abi_sp.reset(new ABISysV_ppc64);
    return g_abi_sp;
  }
  return ABISP();
}

bool ABISysV_ppc64::PrepareTrivialCall(Thread &thread, addr_t sp,
                                       addr_t func_addr, addr_t return_addr,
                                       llvm::ArrayRef<addr_t> args) const {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (log) {
    StreamString s;
    s.Printf("ABISysV_ppc64::PrepareTrivialCall (tid = 0x%" PRIx64
             ", sp = 0x%" PRIx64 ", func_addr = 0x%" PRIx64
             ", return_addr = 0x%" PRIx64,
             thread.GetID(), (uint64_t)sp, (uint64_t)func_addr,
             (uint64_t)return_addr);

    for (size_t i = 0; i < args.size(); ++i)
      s.Printf(", arg%" PRIu64 " = 0x%" PRIx64, static_cast<uint64_t>(i + 1),
               args[i]);
    s.PutCString(")");
    log->PutCString(s.GetString().c_str());
  }

  RegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return false;

  const RegisterInfo *reg_info = nullptr;

  if (args.size() > 8) // TODO handle more than 8 arguments
    return false;

  for (size_t i = 0; i < args.size(); ++i) {
    reg_info = reg_ctx->GetRegisterInfo(eRegisterKindGeneric,
                                        LLDB_REGNUM_GENERIC_ARG1 + i);
    if (log)
      log->Printf("About to write arg%" PRIu64 " (0x%" PRIx64 ") into %s",
                  static_cast<uint64_t>(i + 1), args[i], reg_info->name);
    if (!reg_ctx->WriteRegisterFromUnsigned(reg_info, args[i]))
      return false;
  }

  // First, align the SP

  if (log)
    log->Printf("16-byte aligning SP: 0x%" PRIx64 " to 0x%" PRIx64,
                (uint64_t)sp, (uint64_t)(sp & ~0xfull));

  sp &= ~(0xfull); // 16-byte alignment

  sp -= 8;

  Error error;
  const RegisterInfo *pc_reg_info =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  const RegisterInfo *sp_reg_info =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
  ProcessSP process_sp(thread.GetProcess());

  RegisterValue reg_value;

#if 0
    // This code adds an extra frame so that we don't lose the function that we came from
    // by pushing the PC and the FP and then writing the current FP to point to the FP value
    // we just pushed. It is disabled for now until the stack backtracing code can be debugged.

    // Save current PC
    const RegisterInfo *fp_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP);
    if (reg_ctx->ReadRegister(pc_reg_info, reg_value))
    {
        if (log)
            log->Printf("Pushing the current PC onto the stack: 0x%" PRIx64 ": 0x%" PRIx64, (uint64_t)sp, reg_value.GetAsUInt64());

        if (!process_sp->WritePointerToMemory(sp, reg_value.GetAsUInt64(), error))
            return false;

        sp -= 8;

        // Save current FP
        if (reg_ctx->ReadRegister(fp_reg_info, reg_value))
        {
            if (log)
                log->Printf("Pushing the current FP onto the stack: 0x%" PRIx64 ": 0x%" PRIx64, (uint64_t)sp, reg_value.GetAsUInt64());

            if (!process_sp->WritePointerToMemory(sp, reg_value.GetAsUInt64(), error))
                return false;
        }
        // Setup FP backchain
        reg_value.SetUInt64 (sp);

        if (log)
            log->Printf("Writing FP:  0x%" PRIx64 " (for FP backchain)", reg_value.GetAsUInt64());

        if (!reg_ctx->WriteRegister(fp_reg_info, reg_value))
        {
            return false;
        }

        sp -= 8;
    }
#endif

  if (log)
    log->Printf("Pushing the return address onto the stack: 0x%" PRIx64
                ": 0x%" PRIx64,
                (uint64_t)sp, (uint64_t)return_addr);

  // Save return address onto the stack
  if (!process_sp->WritePointerToMemory(sp, return_addr, error))
    return false;

  // %r1 is set to the actual stack value.

  if (log)
    log->Printf("Writing SP: 0x%" PRIx64, (uint64_t)sp);

  if (!reg_ctx->WriteRegisterFromUnsigned(sp_reg_info, sp))
    return false;

  // %pc is set to the address of the called function.

  if (log)
    log->Printf("Writing IP: 0x%" PRIx64, (uint64_t)func_addr);

  if (!reg_ctx->WriteRegisterFromUnsigned(pc_reg_info, func_addr))
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

  if (current_argument_register < 6) {
    scalar = thread.GetRegisterContext()->ReadRegisterAsUnsigned(
        argument_register_ids[current_argument_register], 0);
    current_argument_register++;
    if (is_signed)
      scalar.SignExtend(bit_width);
  } else {
    uint32_t byte_size = (bit_width + (8 - 1)) / 8;
    Error error;
    if (thread.GetProcess()->ReadScalarIntegerFromMemory(
            current_stack_argument, byte_size, is_signed, scalar, error)) {
      current_stack_argument += byte_size;
      return true;
    }
    return false;
  }
  return true;
}

bool ABISysV_ppc64::GetArgumentValues(Thread &thread, ValueList &values) const {
  unsigned int num_values = values.GetSize();
  unsigned int value_index;

  // Extract the register context so we can read arguments from registers

  RegisterContext *reg_ctx = thread.GetRegisterContext().get();

  if (!reg_ctx)
    return false;

  // Get the pointer to the first stack argument so we have a place to start
  // when reading data

  addr_t sp = reg_ctx->GetSP(0);

  if (!sp)
    return false;

  addr_t current_stack_argument = sp + 48; // jump over return address

  uint32_t argument_register_ids[8];

  argument_register_ids[0] =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1)
          ->kinds[eRegisterKindLLDB];
  argument_register_ids[1] =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG2)
          ->kinds[eRegisterKindLLDB];
  argument_register_ids[2] =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG3)
          ->kinds[eRegisterKindLLDB];
  argument_register_ids[3] =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG4)
          ->kinds[eRegisterKindLLDB];
  argument_register_ids[4] =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG5)
          ->kinds[eRegisterKindLLDB];
  argument_register_ids[5] =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG6)
          ->kinds[eRegisterKindLLDB];
  argument_register_ids[6] =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG7)
          ->kinds[eRegisterKindLLDB];
  argument_register_ids[7] =
      reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG8)
          ->kinds[eRegisterKindLLDB];

  unsigned int current_argument_register = 0;

  for (value_index = 0; value_index < num_values; ++value_index) {
    Value *value = values.GetValueAtIndex(value_index);

    if (!value)
      return false;

    // We currently only support extracting values with Clang QualTypes.
    // Do we care about others?
    CompilerType compiler_type = value->GetCompilerType();
    if (!compiler_type)
      return false;
    bool is_signed;

    if (compiler_type.IsIntegerOrEnumerationType(is_signed)) {
      ReadIntegerArgument(value->GetScalar(), compiler_type.GetBitSize(&thread),
                          is_signed, thread, argument_register_ids,
                          current_argument_register, current_stack_argument);
    } else if (compiler_type.IsPointerType()) {
      ReadIntegerArgument(value->GetScalar(), compiler_type.GetBitSize(&thread),
                          false, thread, argument_register_ids,
                          current_argument_register, current_stack_argument);
    }
  }

  return true;
}

Error ABISysV_ppc64::SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                                          lldb::ValueObjectSP &new_value_sp) {
  Error error;
  if (!new_value_sp) {
    error.SetErrorString("Empty value object for return value.");
    return error;
  }

  CompilerType compiler_type = new_value_sp->GetCompilerType();
  if (!compiler_type) {
    error.SetErrorString("Null clang type for return value.");
    return error;
  }

  Thread *thread = frame_sp->GetThread().get();

  bool is_signed;
  uint32_t count;
  bool is_complex;

  RegisterContext *reg_ctx = thread->GetRegisterContext().get();

  bool set_it_simple = false;
  if (compiler_type.IsIntegerOrEnumerationType(is_signed) ||
      compiler_type.IsPointerType()) {
    const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName("r3", 0);

    DataExtractor data;
    Error data_error;
    size_t num_bytes = new_value_sp->GetData(data, data_error);
    if (data_error.Fail()) {
      error.SetErrorStringWithFormat(
          "Couldn't convert return value to raw data: %s",
          data_error.AsCString());
      return error;
    }
    lldb::offset_t offset = 0;
    if (num_bytes <= 8) {
      uint64_t raw_value = data.GetMaxU64(&offset, num_bytes);

      if (reg_ctx->WriteRegisterFromUnsigned(reg_info, raw_value))
        set_it_simple = true;
    } else {
      error.SetErrorString("We don't support returning longer than 64 bit "
                           "integer values at present.");
    }
  } else if (compiler_type.IsFloatingPointType(count, is_complex)) {
    if (is_complex)
      error.SetErrorString(
          "We don't support returning complex values at present");
    else {
      size_t bit_width = compiler_type.GetBitSize(frame_sp.get());
      if (bit_width <= 64) {
        DataExtractor data;
        Error data_error;
        size_t num_bytes = new_value_sp->GetData(data, data_error);
        if (data_error.Fail()) {
          error.SetErrorStringWithFormat(
              "Couldn't convert return value to raw data: %s",
              data_error.AsCString());
          return error;
        }

        unsigned char buffer[16];
        ByteOrder byte_order = data.GetByteOrder();

        data.CopyByteOrderedData(0, num_bytes, buffer, 16, byte_order);
        set_it_simple = true;
      } else {
        // FIXME - don't know how to do 80 bit long doubles yet.
        error.SetErrorString(
            "We don't support returning float values > 64 bits at present");
      }
    }
  }

  if (!set_it_simple) {
    // Okay we've got a structure or something that doesn't fit in a simple
    // register.
    // We should figure out where it really goes, but we don't support this yet.
    error.SetErrorString("We only support setting simple integer and float "
                         "return types at present.");
  }

  return error;
}

ValueObjectSP ABISysV_ppc64::GetReturnValueObjectSimple(
    Thread &thread, CompilerType &return_compiler_type) const {
  ValueObjectSP return_valobj_sp;
  Value value;

  if (!return_compiler_type)
    return return_valobj_sp;

  // value.SetContext (Value::eContextTypeClangType, return_value_type);
  value.SetCompilerType(return_compiler_type);

  RegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return return_valobj_sp;

  const uint32_t type_flags = return_compiler_type.GetTypeInfo();
  if (type_flags & eTypeIsScalar) {
    value.SetValueType(Value::eValueTypeScalar);

    bool success = false;
    if (type_flags & eTypeIsInteger) {
      // Extract the register context so we can read arguments from registers

      const size_t byte_size = return_compiler_type.GetByteSize(nullptr);
      uint64_t raw_value = thread.GetRegisterContext()->ReadRegisterAsUnsigned(
          reg_ctx->GetRegisterInfoByName("r3", 0), 0);
      const bool is_signed = (type_flags & eTypeIsSigned) != 0;
      switch (byte_size) {
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
        const size_t byte_size = return_compiler_type.GetByteSize(nullptr);
        if (byte_size <= sizeof(long double)) {
          const RegisterInfo *f1_info = reg_ctx->GetRegisterInfoByName("f1", 0);
          RegisterValue f1_value;
          if (reg_ctx->ReadRegister(f1_info, f1_value)) {
            DataExtractor data;
            if (f1_value.GetData(data)) {
              lldb::offset_t offset = 0;
              if (byte_size == sizeof(float)) {
                value.GetScalar() = (float)data.GetFloat(&offset);
                success = true;
              } else if (byte_size == sizeof(double)) {
                value.GetScalar() = (double)data.GetDouble(&offset);
                success = true;
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
    unsigned r3_id =
        reg_ctx->GetRegisterInfoByName("r3", 0)->kinds[eRegisterKindLLDB];
    value.GetScalar() =
        (uint64_t)thread.GetRegisterContext()->ReadRegisterAsUnsigned(r3_id, 0);
    value.SetValueType(Value::eValueTypeScalar);
    return_valobj_sp = ValueObjectConstResult::Create(
        thread.GetStackFrameAtIndex(0).get(), value, ConstString(""));
  } else if (type_flags & eTypeIsVector) {
    const size_t byte_size = return_compiler_type.GetByteSize(nullptr);
    if (byte_size > 0) {
      const RegisterInfo *altivec_reg = reg_ctx->GetRegisterInfoByName("v2", 0);
      if (altivec_reg) {
        if (byte_size <= altivec_reg->byte_size) {
          ProcessSP process_sp(thread.GetProcess());
          if (process_sp) {
            std::unique_ptr<DataBufferHeap> heap_data_ap(
                new DataBufferHeap(byte_size, 0));
            const ByteOrder byte_order = process_sp->GetByteOrder();
            RegisterValue reg_value;
            if (reg_ctx->ReadRegister(altivec_reg, reg_value)) {
              Error error;
              if (reg_value.GetAsMemoryData(
                      altivec_reg, heap_data_ap->GetBytes(),
                      heap_data_ap->GetByteSize(), byte_order, error)) {
                DataExtractor data(DataBufferSP(heap_data_ap.release()),
                                   byte_order, process_sp->GetTarget()
                                                   .GetArchitecture()
                                                   .GetAddressByteSize());
                return_valobj_sp = ValueObjectConstResult::Create(
                    &thread, return_compiler_type, ConstString(""), data);
              }
            }
          }
        }
      }
    }
  }

  return return_valobj_sp;
}

ValueObjectSP ABISysV_ppc64::GetReturnValueObjectImpl(
    Thread &thread, CompilerType &return_compiler_type) const {
  ValueObjectSP return_valobj_sp;

  if (!return_compiler_type)
    return return_valobj_sp;

  ExecutionContext exe_ctx(thread.shared_from_this());
  return_valobj_sp = GetReturnValueObjectSimple(thread, return_compiler_type);
  if (return_valobj_sp)
    return return_valobj_sp;

  RegisterContextSP reg_ctx_sp = thread.GetRegisterContext();
  if (!reg_ctx_sp)
    return return_valobj_sp;

  const size_t bit_width = return_compiler_type.GetBitSize(&thread);
  if (return_compiler_type.IsAggregateType()) {
    Target *target = exe_ctx.GetTargetPtr();
    bool is_memory = true;
    if (bit_width <= 128) {
      ByteOrder target_byte_order = target->GetArchitecture().GetByteOrder();
      DataBufferSP data_sp(new DataBufferHeap(16, 0));
      DataExtractor return_ext(data_sp, target_byte_order,
                               target->GetArchitecture().GetAddressByteSize());

      const RegisterInfo *r3_info = reg_ctx_sp->GetRegisterInfoByName("r3", 0);
      const RegisterInfo *rdx_info =
          reg_ctx_sp->GetRegisterInfoByName("rdx", 0);

      RegisterValue r3_value, rdx_value;
      reg_ctx_sp->ReadRegister(r3_info, r3_value);
      reg_ctx_sp->ReadRegister(rdx_info, rdx_value);

      DataExtractor r3_data, rdx_data;

      r3_value.GetData(r3_data);
      rdx_value.GetData(rdx_data);

      uint32_t fp_bytes =
          0; // Tracks how much of the xmm registers we've consumed so far
      uint32_t integer_bytes =
          0; // Tracks how much of the r3/rds registers we've consumed so far

      const uint32_t num_children = return_compiler_type.GetNumFields();

      // Since we are in the small struct regime, assume we are not in memory.
      is_memory = false;

      for (uint32_t idx = 0; idx < num_children; idx++) {
        std::string name;
        uint64_t field_bit_offset = 0;
        bool is_signed;
        bool is_complex;
        uint32_t count;

        CompilerType field_compiler_type = return_compiler_type.GetFieldAtIndex(
            idx, name, &field_bit_offset, nullptr, nullptr);
        const size_t field_bit_width = field_compiler_type.GetBitSize(&thread);

        // If there are any unaligned fields, this is stored in memory.
        if (field_bit_offset % field_bit_width != 0) {
          is_memory = true;
          break;
        }

        uint32_t field_byte_width = field_bit_width / 8;
        uint32_t field_byte_offset = field_bit_offset / 8;

        DataExtractor *copy_from_extractor = nullptr;
        uint32_t copy_from_offset = 0;

        if (field_compiler_type.IsIntegerOrEnumerationType(is_signed) ||
            field_compiler_type.IsPointerType()) {
          if (integer_bytes < 8) {
            if (integer_bytes + field_byte_width <= 8) {
              // This is in RAX, copy from register to our result structure:
              copy_from_extractor = &r3_data;
              copy_from_offset = integer_bytes;
              integer_bytes += field_byte_width;
            } else {
              // The next field wouldn't fit in the remaining space, so we
              // pushed it to rdx.
              copy_from_extractor = &rdx_data;
              copy_from_offset = 0;
              integer_bytes = 8 + field_byte_width;
            }
          } else if (integer_bytes + field_byte_width <= 16) {
            copy_from_extractor = &rdx_data;
            copy_from_offset = integer_bytes - 8;
            integer_bytes += field_byte_width;
          } else {
            // The last field didn't fit.  I can't see how that would happen w/o
            // the overall size being
            // greater than 16 bytes.  For now, return a nullptr return value
            // object.
            return return_valobj_sp;
          }
        } else if (field_compiler_type.IsFloatingPointType(count, is_complex)) {
          // Structs with long doubles are always passed in memory.
          if (field_bit_width == 128) {
            is_memory = true;
            break;
          } else if (field_bit_width == 64) {
            copy_from_offset = 0;
            fp_bytes += field_byte_width;
          } else if (field_bit_width == 32) {
            // This one is kind of complicated.  If we are in an "eightbyte"
            // with another float, we'll
            // be stuffed into an xmm register with it.  If we are in an
            // "eightbyte" with one or more ints,
            // then we will be stuffed into the appropriate GPR with them.
            bool in_gpr;
            if (field_byte_offset % 8 == 0) {
              // We are at the beginning of one of the eightbytes, so check the
              // next element (if any)
              if (idx == num_children - 1)
                in_gpr = false;
              else {
                uint64_t next_field_bit_offset = 0;
                CompilerType next_field_compiler_type =
                    return_compiler_type.GetFieldAtIndex(idx + 1, name,
                                                         &next_field_bit_offset,
                                                         nullptr, nullptr);
                if (next_field_compiler_type.IsIntegerOrEnumerationType(
                        is_signed))
                  in_gpr = true;
                else {
                  copy_from_offset = 0;
                  in_gpr = false;
                }
              }
            } else if (field_byte_offset % 4 == 0) {
              // We are inside of an eightbyte, so see if the field before us is
              // floating point:
              // This could happen if somebody put padding in the structure.
              if (idx == 0)
                in_gpr = false;
              else {
                uint64_t prev_field_bit_offset = 0;
                CompilerType prev_field_compiler_type =
                    return_compiler_type.GetFieldAtIndex(idx - 1, name,
                                                         &prev_field_bit_offset,
                                                         nullptr, nullptr);
                if (prev_field_compiler_type.IsIntegerOrEnumerationType(
                        is_signed))
                  in_gpr = true;
                else {
                  copy_from_offset = 4;
                  in_gpr = false;
                }
              }
            } else {
              is_memory = true;
              continue;
            }

            // Okay, we've figured out whether we are in GPR or XMM, now figure
            // out which one.
            if (in_gpr) {
              if (integer_bytes < 8) {
                // This is in RAX, copy from register to our result structure:
                copy_from_extractor = &r3_data;
                copy_from_offset = integer_bytes;
                integer_bytes += field_byte_width;
              } else {
                copy_from_extractor = &rdx_data;
                copy_from_offset = integer_bytes - 8;
                integer_bytes += field_byte_width;
              }
            } else {
              fp_bytes += field_byte_width;
            }
          }
        }

        // These two tests are just sanity checks.  If I somehow get the
        // type calculation wrong above it is better to just return nothing
        // than to assert or crash.
        if (!copy_from_extractor)
          return return_valobj_sp;
        if (copy_from_offset + field_byte_width >
            copy_from_extractor->GetByteSize())
          return return_valobj_sp;

        copy_from_extractor->CopyByteOrderedData(
            copy_from_offset, field_byte_width,
            data_sp->GetBytes() + field_byte_offset, field_byte_width,
            target_byte_order);
      }

      if (!is_memory) {
        // The result is in our data buffer.  Let's make a variable object out
        // of it:
        return_valobj_sp = ValueObjectConstResult::Create(
            &thread, return_compiler_type, ConstString(""), return_ext);
      }
    }

    // FIXME: This is just taking a guess, r3 may very well no longer hold the
    // return storage location.
    // If we are going to do this right, when we make a new frame we should
    // check to see if it uses a memory
    // return, and if we are at the first instruction and if so stash away the
    // return location.  Then we would
    // only return the memory return value if we know it is valid.

    if (is_memory) {
      unsigned r3_id =
          reg_ctx_sp->GetRegisterInfoByName("r3", 0)->kinds[eRegisterKindLLDB];
      lldb::addr_t storage_addr =
          (uint64_t)thread.GetRegisterContext()->ReadRegisterAsUnsigned(r3_id,
                                                                        0);
      return_valobj_sp = ValueObjectMemory::Create(
          &thread, "", Address(storage_addr, nullptr), return_compiler_type);
    }
  }

  return return_valobj_sp;
}

bool ABISysV_ppc64::CreateFunctionEntryUnwindPlan(UnwindPlan &unwind_plan) {
  unwind_plan.Clear();
  unwind_plan.SetRegisterKind(eRegisterKindDWARF);

  uint32_t lr_reg_num = dwarf_lr;
  uint32_t sp_reg_num = dwarf_r1;
  uint32_t pc_reg_num = dwarf_pc;

  UnwindPlan::RowSP row(new UnwindPlan::Row);

  // Our Call Frame Address is the stack pointer value
  row->GetCFAValue().SetIsRegisterPlusOffset(sp_reg_num, 0);

  // The previous PC is in the LR
  row->SetRegisterLocationToRegister(pc_reg_num, lr_reg_num, true);
  unwind_plan.AppendRow(row);

  // All other registers are the same.

  unwind_plan.SetSourceName("ppc64 at-func-entry default");
  unwind_plan.SetSourcedFromCompiler(eLazyBoolNo);

  return true;
}

bool ABISysV_ppc64::CreateDefaultUnwindPlan(UnwindPlan &unwind_plan) {
  unwind_plan.Clear();
  unwind_plan.SetRegisterKind(eRegisterKindDWARF);

  uint32_t sp_reg_num = dwarf_r1;
  uint32_t pc_reg_num = dwarf_lr;

  UnwindPlan::RowSP row(new UnwindPlan::Row);

  const int32_t ptr_size = 8;
  row->GetCFAValue().SetIsRegisterDereferenced(sp_reg_num);

  row->SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * 2, true);
  row->SetRegisterLocationToIsCFAPlusOffset(sp_reg_num, 0, true);
  row->SetRegisterLocationToAtCFAPlusOffset(dwarf_cr, ptr_size, true);

  unwind_plan.AppendRow(row);
  unwind_plan.SetSourceName("ppc64 default unwind plan");
  unwind_plan.SetSourcedFromCompiler(eLazyBoolNo);
  unwind_plan.SetUnwindPlanValidAtAllInstructions(eLazyBoolNo);
  unwind_plan.SetReturnAddressRegister(dwarf_lr);
  return true;
}

bool ABISysV_ppc64::RegisterIsVolatile(const RegisterInfo *reg_info) {
  return !RegisterIsCalleeSaved(reg_info);
}

// See "Register Usage" in the
// "System V Application Binary Interface"
// "64-bit PowerPC ELF Application Binary Interface Supplement"
// current version is 1.9 released 2004 at
// http://refspecs.linuxfoundation.org/ELF/ppc64/PPC-elf64abi-1.9.pdf

bool ABISysV_ppc64::RegisterIsCalleeSaved(const RegisterInfo *reg_info) {
  if (reg_info) {
    // Preserved registers are :
    //    r1,r2,r13-r31
    //    cr2-cr4 (partially preserved)
    //    f14-f31 (not yet)
    //    v20-v31 (not yet)
    //    vrsave (not yet)

    const char *name = reg_info->name;
    if (name[0] == 'r') {
      if ((name[1] == '1' || name[1] == '2') && name[2] == '\0')
        return true;
      if (name[1] == '1' && name[2] > '2')
        return true;
      if ((name[1] == '2' || name[1] == '3') && name[2] != '\0')
        return true;
    }

    if (name[0] == 'f' && name[1] >= '0' && name[2] <= '9') {
      if (name[2] == '\0')
        return false;
      if (name[1] == '1' && name[2] >= '4')
        return true;
      if ((name[1] == '2' || name[1] == '3') && name[2] != '\0')
        return true;
    }

    if (name[0] == 's' && name[1] == 'p' && name[2] == '\0') // sp
      return true;
    if (name[0] == 'f' && name[1] == 'p' && name[2] == '\0') // fp
      return true;
    if (name[0] == 'p' && name[1] == 'c' && name[2] == '\0') // pc
      return true;
  }
  return false;
}

void ABISysV_ppc64::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "System V ABI for ppc64 targets", CreateInstance);
}

void ABISysV_ppc64::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString ABISysV_ppc64::GetPluginNameStatic() {
  static ConstString g_name("sysv-ppc64");
  return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------

lldb_private::ConstString ABISysV_ppc64::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ABISysV_ppc64::GetPluginVersion() { return 1; }
