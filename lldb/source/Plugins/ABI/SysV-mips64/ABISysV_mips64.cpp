//===-- ABISysV_mips64.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ABISysV_mips64.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"

using namespace lldb;
using namespace lldb_private;

enum dwarf_regnums
{
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
    dwarf_sr,
    dwarf_lo,
    dwarf_hi,
    dwarf_bad,
    dwarf_cause,
    dwarf_pc
};

static const RegisterInfo
g_register_infos_mips64[] =
{
   //  NAME      ALT    SZ OFF ENCODING        FORMAT         EH_FRAME           DWARF                   GENERIC                     PROCESS PLUGIN          LLDB NATIVE            VALUE REGS INVALIDATE REGS
  //  ========  ======  == === =============  ==========     =============      =================       ====================        =================       ====================    ========== ===============
    { "r0"    , "zero", 8,  0, eEncodingUint, eFormatHex,  {     dwarf_r0,          dwarf_r0,           LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r1"    , "AT",   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r1,          dwarf_r1,           LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r2"    , "v0",   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r2,          dwarf_r2,           LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r3"    , "v1",   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r3,          dwarf_r3,           LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r4"    , "arg1", 8,  0, eEncodingUint, eFormatHex,  {     dwarf_r4,          dwarf_r4,           LLDB_REGNUM_GENERIC_ARG1,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r5"    , "arg2", 8,  0, eEncodingUint, eFormatHex,  {     dwarf_r5,          dwarf_r5,           LLDB_REGNUM_GENERIC_ARG2,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r6"    , "arg3", 8,  0, eEncodingUint, eFormatHex,  {     dwarf_r6,          dwarf_r6,           LLDB_REGNUM_GENERIC_ARG3,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r7"    , "arg4", 8,  0, eEncodingUint, eFormatHex,  {     dwarf_r7,          dwarf_r7,           LLDB_REGNUM_GENERIC_ARG4,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r8"    , "arg5", 8,  0, eEncodingUint, eFormatHex,  {     dwarf_r8,          dwarf_r8,           LLDB_REGNUM_GENERIC_ARG5,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r9"    , "arg6", 8,  0, eEncodingUint, eFormatHex,  {     dwarf_r9,          dwarf_r9,           LLDB_REGNUM_GENERIC_ARG6,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r10"   , "arg7", 8,  0, eEncodingUint, eFormatHex,  {     dwarf_r10,         dwarf_r10,          LLDB_REGNUM_GENERIC_ARG7,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r11"   , "arg8", 8,  0, eEncodingUint, eFormatHex,  {     dwarf_r11,         dwarf_r11,          LLDB_REGNUM_GENERIC_ARG8,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r12"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r12,         dwarf_r12,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r13"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r13,         dwarf_r13,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r14"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r14,         dwarf_r14,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r15"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r15,         dwarf_r15,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r16"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r16,         dwarf_r16,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r17"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r17,         dwarf_r17,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r18"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r18,         dwarf_r18,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r19"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r19,         dwarf_r19,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r20"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r20,         dwarf_r20,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r21"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r21,         dwarf_r21,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r22"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r22,         dwarf_r22,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r23"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r23,         dwarf_r23,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r24"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r24,         dwarf_r24,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r25"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r25,         dwarf_r25,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r26"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r26,         dwarf_r26,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r27"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r27,         dwarf_r27,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r28"   , "gp",   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r28,         dwarf_r28,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r29"   , "sp",   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r29,         dwarf_r29,          LLDB_REGNUM_GENERIC_SP,     LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r30"   , "fp",   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r30,         dwarf_r30,          LLDB_REGNUM_GENERIC_FP,     LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "r31"   , "ra",   8,  0, eEncodingUint, eFormatHex,  {     dwarf_r31,         dwarf_r31,          LLDB_REGNUM_GENERIC_RA,     LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "sr"    , NULL,   4,  0, eEncodingUint, eFormatHex,  {     dwarf_sr,          dwarf_sr,           LLDB_REGNUM_GENERIC_FLAGS,  LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "lo"    , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_lo,          dwarf_lo,           LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "hi"    , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_hi,          dwarf_hi,           LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "bad"   , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_bad,         dwarf_bad,          LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "cause" , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_cause,       dwarf_cause,        LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
    { "pc"    , NULL,   8,  0, eEncodingUint, eFormatHex,  {     dwarf_pc,          dwarf_pc,           LLDB_REGNUM_GENERIC_PC,     LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM },  NULL,      NULL},
};

static const uint32_t k_num_register_infos = llvm::array_lengthof(g_register_infos_mips64);

const lldb_private::RegisterInfo *
ABISysV_mips64::GetRegisterInfoArray (uint32_t &count)
{
    count = k_num_register_infos;
    return g_register_infos_mips64;
}

size_t
ABISysV_mips64::GetRedZoneSize () const
{
    return 0;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
ABISP
ABISysV_mips64::CreateInstance (const ArchSpec &arch)
{
    static ABISP g_abi_sp;
    const llvm::Triple::ArchType arch_type = arch.GetTriple().getArch();
    if ((arch_type == llvm::Triple::mips64) ||
        (arch_type == llvm::Triple::mips64el))
    {
        if (!g_abi_sp)
            g_abi_sp.reset (new ABISysV_mips64);
        return g_abi_sp;
    }
    return ABISP();
}

bool
ABISysV_mips64::PrepareTrivialCall (Thread &thread,
                                  addr_t sp,
                                  addr_t func_addr,
                                  addr_t return_addr,
                                  llvm::ArrayRef<addr_t> args) const
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    if (log)
    {
        StreamString s;
        s.Printf("ABISysV_mips64::PrepareTrivialCall (tid = 0x%" PRIx64 ", sp = 0x%" PRIx64 ", func_addr = 0x%" PRIx64 ", return_addr = 0x%" PRIx64,
                    thread.GetID(),
                    (uint64_t)sp,
                    (uint64_t)func_addr,
                    (uint64_t)return_addr);

        for (size_t i = 0; i < args.size(); ++i)
            s.Printf (", arg%zd = 0x%" PRIx64, i + 1, args[i]);
        s.PutCString (")");
        log->PutCString(s.GetString().c_str());
    }

    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;

    const RegisterInfo *reg_info = NULL;

    if (args.size() > 8) // TODO handle more than 8 arguments
        return false;

    for (size_t i = 0; i < args.size(); ++i)
    {
        reg_info = reg_ctx->GetRegisterInfo(eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1 + i);
        if (log)
            log->Printf("About to write arg%zd (0x%" PRIx64 ") into %s", i + 1, args[i], reg_info->name);
        if (!reg_ctx->WriteRegisterFromUnsigned(reg_info, args[i]))
            return false;
    }

    // First, align the SP

    if (log)
        log->Printf("16-byte aligning SP: 0x%" PRIx64 " to 0x%" PRIx64, (uint64_t)sp, (uint64_t)(sp & ~0xfull));

    sp &= ~(0xfull); // 16-byte alignment

    Error error;
    const RegisterInfo *pc_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    const RegisterInfo *sp_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    const RegisterInfo *ra_reg_info = reg_ctx->GetRegisterInfo (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA);
    const RegisterInfo *r25_info = reg_ctx->GetRegisterInfoByName("r25", 0);

    if (log)
    log->Printf("Writing SP: 0x%" PRIx64, (uint64_t)sp);

    // Set "sp" to the requested value
    if (!reg_ctx->WriteRegisterFromUnsigned (sp_reg_info, sp))
        return false;

    if (log)
    log->Printf("Writing RA: 0x%" PRIx64, (uint64_t)return_addr);

    // Set "ra" to the return address
    if (!reg_ctx->WriteRegisterFromUnsigned (ra_reg_info, return_addr))
        return false;

    if (log)
        log->Printf("Writing PC: 0x%" PRIx64, (uint64_t)func_addr);

    // Set pc to the address of the called function.
    if (!reg_ctx->WriteRegisterFromUnsigned (pc_reg_info, func_addr))
        return false;

    if (log)
        log->Printf("Writing r25: 0x%" PRIx64, (uint64_t)func_addr);

    // All callers of position independent functions must place the address of the called function in t9 (r25)
    if (!reg_ctx->WriteRegisterFromUnsigned (r25_info, func_addr))
        return false;

    return true;
}

bool
ABISysV_mips64::GetArgumentValues (Thread &thread, ValueList &values) const
{
    return false;
}

Error
ABISysV_mips64::SetReturnValueObject(lldb::StackFrameSP &frame_sp, lldb::ValueObjectSP &new_value_sp)
{
    Error error;
    if (!new_value_sp)
    {
        error.SetErrorString("Empty value object for return value.");
        return error;
    }

    CompilerType compiler_type = new_value_sp->GetCompilerType();
    if (!compiler_type)
    {
        error.SetErrorString ("Null clang type for return value.");
        return error;
    }

    Thread *thread = frame_sp->GetThread().get();

    RegisterContext *reg_ctx = thread->GetRegisterContext().get();

    if (!reg_ctx)
        error.SetErrorString("no registers are available");
        
    DataExtractor data;
    Error data_error;
    size_t num_bytes = new_value_sp->GetData(data, data_error);
    if (data_error.Fail())
    {
        error.SetErrorStringWithFormat("Couldn't convert return value to raw data: %s", data_error.AsCString());
        return error;
    }

    const uint32_t type_flags = compiler_type.GetTypeInfo (NULL);
    
    if (type_flags & eTypeIsScalar ||
        type_flags & eTypeIsPointer)
    {
        if (type_flags & eTypeIsInteger ||
            type_flags & eTypeIsPointer )
        {
            lldb::offset_t offset = 0;
            
            if (num_bytes <= 16)
            {
                const RegisterInfo *r2_info = reg_ctx->GetRegisterInfoByName("r2", 0);
                if (num_bytes <= 8)
                {
                    uint64_t raw_value = data.GetMaxU64(&offset, num_bytes);
                        
                    if (!reg_ctx->WriteRegisterFromUnsigned (r2_info, raw_value))
                        error.SetErrorString ("failed to write register r2");
                }
                else
                {
                    uint64_t raw_value = data.GetMaxU64(&offset, 8);
                    if (reg_ctx->WriteRegisterFromUnsigned (r2_info, raw_value))
                    {
                        const RegisterInfo *r3_info = reg_ctx->GetRegisterInfoByName("r3", 0);
                        raw_value = data.GetMaxU64(&offset, num_bytes - offset);
                        
                        if (!reg_ctx->WriteRegisterFromUnsigned (r3_info, raw_value))
                            error.SetErrorString ("failed to write register r3");
                    }
                    else
                        error.SetErrorString ("failed to write register r2");
                }
            }
            else
            {
                error.SetErrorString("We don't support returning longer than 128 bit integer values at present.");
            }
        }
        else if (type_flags & eTypeIsFloat)
        {
            error.SetErrorString("TODO: Handle Float Types.");
        }
    }
    else if (type_flags & eTypeIsVector)
    {
        error.SetErrorString("returning vector values are not supported");
    }

    return error;
}


ValueObjectSP
ABISysV_mips64::GetReturnValueObjectSimple (Thread &thread, CompilerType &return_compiler_type) const
{
    ValueObjectSP return_valobj_sp;
    return return_valobj_sp;
}

ValueObjectSP
ABISysV_mips64::GetReturnValueObjectImpl (Thread &thread, CompilerType &return_compiler_type) const
{
    ValueObjectSP return_valobj_sp;
    Value value;
    Error error;
    
    ExecutionContext exe_ctx (thread.shared_from_this());
    if (exe_ctx.GetTargetPtr() == NULL || exe_ctx.GetProcessPtr() == NULL)
        return return_valobj_sp;

    value.SetCompilerType(return_compiler_type);

    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return return_valobj_sp;

    Target *target = exe_ctx.GetTargetPtr();
    ByteOrder target_byte_order = target->GetArchitecture().GetByteOrder();
    const size_t byte_size = return_compiler_type.GetByteSize(nullptr);
    const uint32_t type_flags = return_compiler_type.GetTypeInfo (NULL);
    
    const RegisterInfo *r2_info = reg_ctx->GetRegisterInfoByName("r2", 0);
    const RegisterInfo *r3_info = reg_ctx->GetRegisterInfoByName("r3", 0);

    if (type_flags & eTypeIsScalar ||
        type_flags & eTypeIsPointer)
    {
        value.SetValueType(Value::eValueTypeScalar);

        bool success = false;
        if (type_flags & eTypeIsInteger ||
            type_flags & eTypeIsPointer)
        {
            // Extract the register context so we can read arguments from registers
            // In MIPS register "r2" (v0) holds the integer function return values

            uint64_t raw_value = reg_ctx->ReadRegisterAsUnsigned(r2_info, 0);

            const bool is_signed = (type_flags & eTypeIsSigned) != 0;
            switch (byte_size)
            {
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
        }
        else if (type_flags & eTypeIsFloat)
        {
            if (type_flags & eTypeIsComplex)
            {
                // Don't handle complex yet.
            }
            else
            {
                if (byte_size <= sizeof(long double))
                {
                    const RegisterInfo *f0_info = reg_ctx->GetRegisterInfoByName("f0", 0);
                    const RegisterInfo *f2_info = reg_ctx->GetRegisterInfoByName("f2", 0);
                    RegisterValue f0_value, f2_value;
                    DataExtractor f0_data, f2_data;
                    
                    reg_ctx->ReadRegister (f0_info, f0_value);
                    reg_ctx->ReadRegister (f2_info, f2_value);
                    
                    f0_value.GetData(f0_data);
                    f2_value.GetData(f2_data);

                    lldb::offset_t offset = 0;
                    if (byte_size == sizeof(float))
                    {
                        value.GetScalar() = (float) f0_data.GetFloat(&offset);
                        success = true;
                    }
                    else if (byte_size == sizeof(double))
                    {
                        value.GetScalar() = (double) f0_data.GetDouble(&offset);
                        success = true;
                    }
                    else if (byte_size == sizeof(long double))
                    {
                        DataExtractor *copy_from_extractor = NULL;
                        DataBufferSP data_sp (new DataBufferHeap(16, 0));
                        DataExtractor return_ext (data_sp, 
                                                  target_byte_order, 
                                                  target->GetArchitecture().GetAddressByteSize());

                        if (target_byte_order == eByteOrderLittle)
                        {
                             f0_data.Append(f2_data);
                             copy_from_extractor = &f0_data;
                        }
                        else
                        {
                            f2_data.Append(f0_data);
                            copy_from_extractor = &f2_data;
                        }

                        copy_from_extractor->CopyByteOrderedData (0,
                                                                  byte_size, 
                                                                  data_sp->GetBytes(),
                                                                  byte_size, 
                                                                  target_byte_order);

                        return_valobj_sp = ValueObjectConstResult::Create (&thread, 
                                                                           return_compiler_type,
                                                                           ConstString(""),
                                                                           return_ext);
                        return return_valobj_sp;

                    }
                }
            }
        }

        if (success)
        return_valobj_sp = ValueObjectConstResult::Create (thread.GetStackFrameAtIndex(0).get(),
                                                           value,
                                                           ConstString(""));
    }
    else if (type_flags & eTypeIsStructUnion ||
             type_flags & eTypeIsClass ||
             type_flags & eTypeIsVector)
    {
        // Any structure of up to 16 bytes in size is returned in the registers.
        if (byte_size <= 16)
        {
            DataBufferSP data_sp (new DataBufferHeap(16, 0));
            DataExtractor return_ext (data_sp, 
                                      target_byte_order, 
                                      target->GetArchitecture().GetAddressByteSize());

            RegisterValue r2_value, r3_value, f0_value, f1_value, f2_value;

            uint32_t integer_bytes = 0;         // Tracks how much bytes of r2 and r3 registers we've consumed so far
            bool use_fp_regs = 0;               // True if return values are in FP return registers.
            bool found_non_fp_field = 0;        // True if we found any non floating point field in structure.
            bool use_r2 = 0;                    // True if return values are in r2 register.
            bool use_r3 = 0;                    // True if return values are in r3 register.
            bool sucess = 0;                    // True if the result is copied into our data buffer
            std::string name;
            bool is_complex;
            uint32_t count;
            const uint32_t num_children = return_compiler_type.GetNumFields ();

            // A structure consisting of one or two FP values (and nothing else) will be
            // returned in the two FP return-value registers i.e fp0 and fp2.
            if (num_children <= 2)
            {
                uint64_t field_bit_offset = 0;

                // Check if this structure contains only floating point fields
                for (uint32_t idx = 0; idx < num_children; idx++)
                {
                    CompilerType field_compiler_type = return_compiler_type.GetFieldAtIndex (idx, name, &field_bit_offset, NULL, NULL);
                    
                    if (field_compiler_type.IsFloatingPointType (count, is_complex))
                        use_fp_regs = 1;
                    else
                        found_non_fp_field = 1;
                }

                if (use_fp_regs && !found_non_fp_field)
                {
                    // We have one or two FP-only values in this structure. Get it from f0/f2 registers.
                    DataExtractor f0_data, f1_data, f2_data;
                    const RegisterInfo *f0_info = reg_ctx->GetRegisterInfoByName("f0", 0);
                    const RegisterInfo *f1_info = reg_ctx->GetRegisterInfoByName("f1", 0);
                    const RegisterInfo *f2_info = reg_ctx->GetRegisterInfoByName("f2", 0);

                    reg_ctx->ReadRegister (f0_info, f0_value);
                    reg_ctx->ReadRegister (f2_info, f2_value);

                    f0_value.GetData(f0_data);
                    f2_value.GetData(f2_data);

                    for (uint32_t idx = 0; idx < num_children; idx++)
                    {
                        CompilerType field_compiler_type = return_compiler_type.GetFieldAtIndex (idx, name, &field_bit_offset, NULL, NULL);
                        const size_t field_byte_width = field_compiler_type.GetByteSize(nullptr);

                        DataExtractor *copy_from_extractor = NULL;

                        if (idx == 0)
                        {
                            if (field_byte_width == 16)                 // This case is for long double type.
                            {
                                // If structure contains long double type, then it is returned in fp0/fp1 registers.
                                reg_ctx->ReadRegister (f1_info, f1_value);
                                f1_value.GetData(f1_data);
                                
                                if (target_byte_order == eByteOrderLittle)
                                {
                                    f0_data.Append(f1_data);
                                    copy_from_extractor = &f0_data;
                                }
                                else
                                {
                                    f1_data.Append(f0_data);
                                    copy_from_extractor = &f1_data;
                                }
                            }
                            else
                                copy_from_extractor = &f0_data;        // This is in f0, copy from register to our result structure
                        }
                        else
                            copy_from_extractor = &f2_data;        // This is in f2, copy from register to our result structure

                        // Sanity check to avoid crash
                        if (!copy_from_extractor || field_byte_width > copy_from_extractor->GetByteSize())
                            return return_valobj_sp;

                        // copy the register contents into our data buffer
                        copy_from_extractor->CopyByteOrderedData (0,
                                                                  field_byte_width, 
                                                                  data_sp->GetBytes() + (field_bit_offset/8),
                                                                  field_byte_width, 
                                                                  target_byte_order);
                    }

                    // The result is in our data buffer.  Create a variable object out of it
                    return_valobj_sp = ValueObjectConstResult::Create (&thread, 
                                                                       return_compiler_type,
                                                                       ConstString(""),
                                                                       return_ext);

                    return return_valobj_sp;
                }
            }

            // If we reach here, it means this structure either contains more than two fields or 
            // it contains at least one non floating point type.
            // In that case, all fields are returned in GP return registers.
            for (uint32_t idx = 0; idx < num_children; idx++)
            {
                uint64_t field_bit_offset = 0;
                bool is_signed;
                uint32_t padding;

                CompilerType field_compiler_type = return_compiler_type.GetFieldAtIndex (idx, name, &field_bit_offset, NULL, NULL);
                const size_t field_byte_width = field_compiler_type.GetByteSize(nullptr);

                // if we don't know the size of the field (e.g. invalid type), just bail out
                if (field_byte_width == 0)
                    break;

                uint32_t field_byte_offset = field_bit_offset/8;

                if (field_compiler_type.IsIntegerType (is_signed)
                    || field_compiler_type.IsPointerType ()
                    || field_compiler_type.IsFloatingPointType (count, is_complex))
                {
                    padding = field_byte_offset - integer_bytes;

                    if (integer_bytes < 8)
                    {
                        // We have not yet consumed r2 completely.
                        if (integer_bytes + field_byte_width + padding <= 8)
                        {
                            // This field fits in r2, copy its value from r2 to our result structure
                            integer_bytes = integer_bytes + field_byte_width + padding;  // Increase the consumed bytes.
                            use_r2 = 1;
                        }
                        else
                        {
                            // There isn't enough space left in r2 for this field, so this will be in r3.
                            integer_bytes = integer_bytes + field_byte_width + padding;  // Increase the consumed bytes.
                            use_r3 = 1;
                        }
                    }
                    // We already have consumed at-least 8 bytes that means r2 is done, and this field will be in r3.
                    // Check if this field can fit in r3.
                    else if (integer_bytes + field_byte_width + padding <= 16)
                    {
                        integer_bytes = integer_bytes + field_byte_width + padding;
                        use_r3 = 1;
                    }
                    else
                    {
                        // There isn't any space left for this field, this should not happen as we have already checked
                        // the overall size is not greater than 16 bytes. For now, return a NULL return value object.
                        return return_valobj_sp;
                    }
                }
            }
            // Vector types upto 16 bytes are returned in GP return registers
            if (type_flags & eTypeIsVector)
            {
                if (byte_size <= 8)
                    use_r2 = 1;
                else
                {
                    use_r2 = 1;
                    use_r3 = 1;
                }    
            }

            if (use_r2)
            {
                reg_ctx->ReadRegister (r2_info, r2_value);

                const size_t bytes_copied = r2_value.GetAsMemoryData (r2_info,
                                                                      data_sp->GetBytes(),
                                                                      r2_info->byte_size,
                                                                      target_byte_order,
                                                                      error);
                if (bytes_copied != r2_info->byte_size)
                    return return_valobj_sp;
                sucess = 1;
            }
            if (use_r3)
            {
                reg_ctx->ReadRegister (r3_info, r3_value);
                const size_t bytes_copied = r3_value.GetAsMemoryData (r3_info,
                                                                      data_sp->GetBytes() + r2_info->byte_size,
                                                                      r3_info->byte_size,
                                                                      target_byte_order,
                                                                      error);                                                       
                                                        
                if (bytes_copied != r3_info->byte_size)
                    return return_valobj_sp;
                sucess = 1;
            }
            if (sucess)
            {
                // The result is in our data buffer.  Create a variable object out of it
                return_valobj_sp = ValueObjectConstResult::Create (&thread, 
                                                                   return_compiler_type,
                                                                   ConstString(""),
                                                                   return_ext);
            }
            return return_valobj_sp;
        }

        // Any structure/vector greater than 16 bytes in size is returned in memory.
        // The pointer to that memory is returned in r2.
        uint64_t mem_address = reg_ctx->ReadRegisterAsUnsigned(reg_ctx->GetRegisterInfoByName("r2", 0), 0);

        // We have got the address. Create a memory object out of it
        return_valobj_sp = ValueObjectMemory::Create (&thread,
                                                      "",
                                                      Address (mem_address, NULL),
                                                      return_compiler_type);
    }
    return return_valobj_sp;
}

bool
ABISysV_mips64::CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    UnwindPlan::RowSP row(new UnwindPlan::Row);

    // Our Call Frame Address is the stack pointer value
    row->GetCFAValue().SetIsRegisterPlusOffset(dwarf_r29, 0);

    // The previous PC is in the RA
    row->SetRegisterLocationToRegister(dwarf_pc, dwarf_r31, true);
    unwind_plan.AppendRow (row);

    // All other registers are the same.

    unwind_plan.SetSourceName ("mips64 at-func-entry default");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetReturnAddressRegister(dwarf_r31);
    return true;
}

bool
ABISysV_mips64::CreateDefaultUnwindPlan (UnwindPlan &unwind_plan)
{
    unwind_plan.Clear();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);

    UnwindPlan::RowSP row(new UnwindPlan::Row);

    row->GetCFAValue().SetIsRegisterPlusOffset(dwarf_r29, 0);

    row->SetRegisterLocationToRegister(dwarf_pc, dwarf_r31, true);

    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("mips64 default unwind plan");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    return true;
}

bool
ABISysV_mips64::RegisterIsVolatile (const RegisterInfo *reg_info)
{
    return !RegisterIsCalleeSaved (reg_info);
}

bool
ABISysV_mips64::RegisterIsCalleeSaved (const RegisterInfo *reg_info)
{
    if (reg_info)
    {
        // Preserved registers are :
        // r16-r23, r28, r29, r30, r31

        int reg = ((reg_info->byte_offset) / 8);

        bool save  = (reg >= 16) && (reg <= 23);
             save |= (reg >= 28) && (reg <= 31);

        return save;
    }
    return false;
}

void
ABISysV_mips64::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "System V ABI for mips64 targets",
                                   CreateInstance);
}

void
ABISysV_mips64::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

lldb_private::ConstString
ABISysV_mips64::GetPluginNameStatic()
{
    static ConstString g_name("sysv-mips64");
    return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
ABISysV_mips64::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ABISysV_mips64::GetPluginVersion()
{
    return 1;
}
