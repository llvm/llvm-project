//===-- ABIMacOSX_i386.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ABIMacOSX_i386.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "llvm/ADT/Triple.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

static const char *pluginName = "ABIMacOSX_i386";
static const char *pluginDesc = "Mac OS X ABI for i386 targets";
static const char *pluginShort = "abi.macosx-i386";

enum
{
    gcc_eax = 0,
    gcc_ecx,
    gcc_edx,
    gcc_ebx,
    gcc_ebp,
    gcc_esp,
    gcc_esi,
    gcc_edi,
    gcc_eip,
    gcc_eflags
};

enum
{
    dwarf_eax = 0,
    dwarf_ecx,
    dwarf_edx,
    dwarf_ebx,
    dwarf_esp,
    dwarf_ebp,
    dwarf_esi,
    dwarf_edi,
    dwarf_eip,
    dwarf_eflags,
    dwarf_stmm0 = 11,
    dwarf_stmm1,
    dwarf_stmm2,
    dwarf_stmm3,
    dwarf_stmm4,
    dwarf_stmm5,
    dwarf_stmm6,
    dwarf_stmm7,
    dwarf_xmm0 = 21,
    dwarf_xmm1,
    dwarf_xmm2,
    dwarf_xmm3,
    dwarf_xmm4,
    dwarf_xmm5,
    dwarf_xmm6,
    dwarf_xmm7,
    dwarf_ymm0 = dwarf_xmm0,
    dwarf_ymm1 = dwarf_xmm1,
    dwarf_ymm2 = dwarf_xmm2,
    dwarf_ymm3 = dwarf_xmm3,
    dwarf_ymm4 = dwarf_xmm4,
    dwarf_ymm5 = dwarf_xmm5,
    dwarf_ymm6 = dwarf_xmm6,
    dwarf_ymm7 = dwarf_xmm7
};

enum
{
    gdb_eax        =  0,
    gdb_ecx        =  1,
    gdb_edx        =  2,
    gdb_ebx        =  3,
    gdb_esp        =  4,
    gdb_ebp        =  5,
    gdb_esi        =  6,
    gdb_edi        =  7,
    gdb_eip        =  8,
    gdb_eflags     =  9,
    gdb_cs         = 10,
    gdb_ss         = 11,
    gdb_ds         = 12,
    gdb_es         = 13,
    gdb_fs         = 14,
    gdb_gs         = 15,
    gdb_stmm0      = 16,
    gdb_stmm1      = 17,
    gdb_stmm2      = 18,
    gdb_stmm3      = 19,
    gdb_stmm4      = 20,
    gdb_stmm5      = 21,
    gdb_stmm6      = 22,
    gdb_stmm7      = 23,
    gdb_fctrl      = 24,    gdb_fcw     = gdb_fctrl,
    gdb_fstat      = 25,    gdb_fsw     = gdb_fstat,
    gdb_ftag       = 26,    gdb_ftw     = gdb_ftag,
    gdb_fiseg      = 27,    gdb_fpu_cs  = gdb_fiseg,
    gdb_fioff      = 28,    gdb_ip      = gdb_fioff,
    gdb_foseg      = 29,    gdb_fpu_ds  = gdb_foseg,
    gdb_fooff      = 30,    gdb_dp      = gdb_fooff,
    gdb_fop        = 31,
    gdb_xmm0       = 32,
    gdb_xmm1       = 33,
    gdb_xmm2       = 34,
    gdb_xmm3       = 35,
    gdb_xmm4       = 36,
    gdb_xmm5       = 37,
    gdb_xmm6       = 38,
    gdb_xmm7       = 39,
    gdb_mxcsr      = 40,
    gdb_mm0        = 41,
    gdb_mm1        = 42,
    gdb_mm2        = 43,
    gdb_mm3        = 44,
    gdb_mm4        = 45,
    gdb_mm5        = 46,
    gdb_mm6        = 47,
    gdb_mm7        = 48,
    gdb_ymm0       = gdb_xmm0,
    gdb_ymm1       = gdb_xmm1,
    gdb_ymm2       = gdb_xmm2,
    gdb_ymm3       = gdb_xmm3,
    gdb_ymm4       = gdb_xmm4,
    gdb_ymm5       = gdb_xmm5,
    gdb_ymm6       = gdb_xmm6,
    gdb_ymm7       = gdb_xmm7
};


static RegisterInfo g_register_infos[] = 
{
  //  NAME      ALT      SZ OFF ENCODING         FORMAT                COMPILER              DWARF                 GENERIC                      GDB                   LLDB NATIVE            VALUE REGS    INVALIDATE REGS
  //  ======    =======  == === =============    ============          ===================== ===================== ============================ ====================  ====================== ==========    ===============
    { "eax",    NULL,    4,  0, eEncodingUint  , eFormatHex          , { gcc_eax             , dwarf_eax           , LLDB_INVALID_REGNUM       , gdb_eax            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ebx"   , NULL,    4,  0, eEncodingUint  , eFormatHex          , { gcc_ebx             , dwarf_ebx           , LLDB_INVALID_REGNUM       , gdb_ebx            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ecx"   , NULL,    4,  0, eEncodingUint  , eFormatHex          , { gcc_ecx             , dwarf_ecx           , LLDB_REGNUM_GENERIC_ARG4  , gdb_ecx            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "edx"   , NULL,    4,  0, eEncodingUint  , eFormatHex          , { gcc_edx             , dwarf_edx           , LLDB_REGNUM_GENERIC_ARG3  , gdb_edx            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "esi"   , NULL,    4,  0, eEncodingUint  , eFormatHex          , { gcc_esi             , dwarf_esi           , LLDB_REGNUM_GENERIC_ARG2  , gdb_esi            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "edi"   , NULL,    4,  0, eEncodingUint  , eFormatHex          , { gcc_edi             , dwarf_edi           , LLDB_REGNUM_GENERIC_ARG1  , gdb_edi            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ebp"   , "fp",    4,  0, eEncodingUint  , eFormatHex          , { gcc_ebp             , dwarf_ebp           , LLDB_REGNUM_GENERIC_FP    , gdb_ebp            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "esp"   , "sp",    4,  0, eEncodingUint  , eFormatHex          , { gcc_esp             , dwarf_esp           , LLDB_REGNUM_GENERIC_SP    , gdb_esp            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "eip"   , "pc",    4,  0, eEncodingUint  , eFormatHex          , { gcc_eip             , dwarf_eip           , LLDB_REGNUM_GENERIC_PC    , gdb_eip            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "eflags", NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_REGNUM_GENERIC_FLAGS , gdb_eflags         , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "cs"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_cs             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ss"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_ss             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ds"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_ds             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "es"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_es             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fs"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fs             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "gs"    , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_gs             , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm0" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_stmm0         , LLDB_INVALID_REGNUM       , gdb_stmm0          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm1" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_stmm1         , LLDB_INVALID_REGNUM       , gdb_stmm1          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm2" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_stmm2         , LLDB_INVALID_REGNUM       , gdb_stmm2          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm3" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_stmm3         , LLDB_INVALID_REGNUM       , gdb_stmm3          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm4" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_stmm4         , LLDB_INVALID_REGNUM       , gdb_stmm4          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm5" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_stmm5         , LLDB_INVALID_REGNUM       , gdb_stmm5          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm6" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_stmm6         , LLDB_INVALID_REGNUM       , gdb_stmm6          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "stmm7" , NULL,   10,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_stmm7         , LLDB_INVALID_REGNUM       , gdb_stmm7          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fctrl" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fctrl          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fstat" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fstat          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ftag"  , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_ftag           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fiseg" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fiseg          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fioff" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fioff          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "foseg" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_foseg          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fooff" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fooff          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "fop"   , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_fop            , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm0"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_xmm0          , LLDB_INVALID_REGNUM       , gdb_xmm0           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm1"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_xmm1          , LLDB_INVALID_REGNUM       , gdb_xmm1           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm2"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_xmm2          , LLDB_INVALID_REGNUM       , gdb_xmm2           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm3"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_xmm3          , LLDB_INVALID_REGNUM       , gdb_xmm3           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm4"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_xmm4          , LLDB_INVALID_REGNUM       , gdb_xmm4           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm5"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_xmm5          , LLDB_INVALID_REGNUM       , gdb_xmm5           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm6"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_xmm6          , LLDB_INVALID_REGNUM       , gdb_xmm6           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "xmm7"  , NULL,   16,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_xmm7          , LLDB_INVALID_REGNUM       , gdb_xmm7           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "mxcsr" , NULL,    4,  0, eEncodingUint  , eFormatHex          , { LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM , LLDB_INVALID_REGNUM       , gdb_mxcsr          , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm0"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_ymm0          , LLDB_INVALID_REGNUM       , gdb_ymm0           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm1"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_ymm1          , LLDB_INVALID_REGNUM       , gdb_ymm1           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm2"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_ymm2          , LLDB_INVALID_REGNUM       , gdb_ymm2           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm3"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_ymm3          , LLDB_INVALID_REGNUM       , gdb_ymm3           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm4"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_ymm4          , LLDB_INVALID_REGNUM       , gdb_ymm4           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm5"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_ymm5          , LLDB_INVALID_REGNUM       , gdb_ymm5           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm6"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_ymm6          , LLDB_INVALID_REGNUM       , gdb_ymm6           , LLDB_INVALID_REGNUM },      NULL,              NULL},
    { "ymm7"  , NULL,   32,  0, eEncodingVector, eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM , dwarf_ymm7          , LLDB_INVALID_REGNUM       , gdb_ymm7           , LLDB_INVALID_REGNUM },      NULL,              NULL}
};

static const uint32_t k_num_register_infos = sizeof(g_register_infos)/sizeof(RegisterInfo);
static bool g_register_info_names_constified = false;

const lldb_private::RegisterInfo *
ABIMacOSX_i386::GetRegisterInfoArray (uint32_t &count)
{
    // Make the C-string names and alt_names for the register infos into const 
    // C-string values by having the ConstString unique the names in the global
    // constant C-string pool.
    if (!g_register_info_names_constified)
    {
        g_register_info_names_constified = true;
        for (uint32_t i=0; i<k_num_register_infos; ++i)
        {
            if (g_register_infos[i].name)
                g_register_infos[i].name = ConstString(g_register_infos[i].name).GetCString();
            if (g_register_infos[i].alt_name)
                g_register_infos[i].alt_name = ConstString(g_register_infos[i].alt_name).GetCString();
        }
    }
    count = k_num_register_infos;
    return g_register_infos;
}

size_t
ABIMacOSX_i386::GetRedZoneSize () const
{
    return 0;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
ABISP
ABIMacOSX_i386::CreateInstance (const ArchSpec &arch)
{
    static ABISP g_abi_sp;
    if (arch.GetTriple().getArch() == llvm::Triple::x86)
    {
        if (!g_abi_sp)
            g_abi_sp.reset (new ABIMacOSX_i386);
        return g_abi_sp;
    }
    return ABISP();
}

bool
ABIMacOSX_i386::PrepareTrivialCall (Thread &thread, 
                                    addr_t sp, 
                                    addr_t func_addr, 
                                    addr_t return_addr, 
                                    addr_t *arg1_ptr,
                                    addr_t *arg2_ptr,
                                    addr_t *arg3_ptr,
                                    addr_t *arg4_ptr,
                                    addr_t *arg5_ptr,
                                    addr_t *arg6_ptr) const
{
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;    
    uint32_t pc_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    uint32_t sp_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    
    // When writing a register value down to memory, the register info used 
    // to write memory just needs to have the correct size of a 32 bit register, 
    // the actual register it pertains to is not important, just the size needs 
    // to be correct. Here we use "eax"...
    const RegisterInfo *reg_info_32 = reg_ctx->GetRegisterInfoByName("eax");
    if (!reg_info_32)
        return false; // TODO this should actually never happen

    // Make room for the argument(s) on the stack

    Error error;
    RegisterValue reg_value;
    
    // Write any arguments onto the stack
    if (arg1_ptr)
    {
        sp -= 4;
        if (arg2_ptr)
        {
            sp -= 4;
            if (arg3_ptr)
            {
                sp -= 4;
                if (arg4_ptr)
                {
                    sp -= 4;
                    if (arg5_ptr)
                    {
                        sp -= 4;
                        if (arg6_ptr)
                        {
                            sp -= 4;
                        }
                    }
                }
            }
        }
    }

    // Align the SP    
    sp &= ~(16ull-1ull); // 16-byte alignment
    
    if (arg1_ptr)
    {
        reg_value.SetUInt32(*arg1_ptr);
        error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                     sp, 
                                                     reg_info_32->byte_size, 
                                                     reg_value);
        if (error.Fail())
            return false;

        if (arg2_ptr)
        {
            reg_value.SetUInt32(*arg2_ptr);
            // The register info used to write memory just needs to have the correct
            // size of a 32 bit register, the actual register it pertains to is not
            // important, just the size needs to be correct. Here we use "eax"...
            error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                         sp + 4, 
                                                         reg_info_32->byte_size, 
                                                         reg_value);
            if (error.Fail())
                return false;
            
            if (arg3_ptr)
            {
                reg_value.SetUInt32(*arg3_ptr);
                // The register info used to write memory just needs to have the correct
                // size of a 32 bit register, the actual register it pertains to is not
                // important, just the size needs to be correct. Here we use "eax"...
                error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                             sp + 8, 
                                                             reg_info_32->byte_size, 
                                                             reg_value);
                if (error.Fail())
                    return false;

                if (arg4_ptr)
                {
                    reg_value.SetUInt32(*arg4_ptr);
                    // The register info used to write memory just needs to have the correct
                    // size of a 32 bit register, the actual register it pertains to is not
                    // important, just the size needs to be correct. Here we use "eax"...
                    error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                                 sp + 12, 
                                                                 reg_info_32->byte_size, 
                                                                 reg_value);
                    if (error.Fail())
                        return false;
                    if (arg5_ptr)
                    {
                        reg_value.SetUInt32(*arg5_ptr);
                        // The register info used to write memory just needs to have the correct
                        // size of a 32 bit register, the actual register it pertains to is not
                        // important, just the size needs to be correct. Here we use "eax"...
                        error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                                     sp + 16, 
                                                                     reg_info_32->byte_size, 
                                                                     reg_value);
                        if (error.Fail())
                            return false;
                        if (arg6_ptr)
                        {
                            reg_value.SetUInt32(*arg6_ptr);
                            // The register info used to write memory just needs to have the correct
                            // size of a 32 bit register, the actual register it pertains to is not
                            // important, just the size needs to be correct. Here we use "eax"...
                            error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                                         sp + 20, 
                                                                         reg_info_32->byte_size, 
                                                                         reg_value);
                            if (error.Fail())
                                return false;
                        }
                    }
                }
            }
        }
    }
    
    
    // The return address is pushed onto the stack (yes after we just set the
    // alignment above!).
    sp -= 4;
    reg_value.SetUInt32(return_addr);
    error = reg_ctx->WriteRegisterValueToMemory (reg_info_32, 
                                                 sp, 
                                                 reg_info_32->byte_size, 
                                                 reg_value);
    if (error.Fail())
        return false;
    
    // %esp is set to the actual stack value.
    
    if (!reg_ctx->WriteRegisterFromUnsigned (sp_reg_num, sp))
        return false;
    
    // %eip is set to the address of the called function.
    
    if (!reg_ctx->WriteRegisterFromUnsigned (pc_reg_num, func_addr))
        return false;
    
    return true;
}

bool
ABIMacOSX_i386::PrepareNormalCall (Thread &thread,
                                   addr_t sp,
                                   addr_t func_addr,
                                   addr_t return_addr,
                                   ValueList &args) const
{
    ExecutionContext exe_ctx (thread.shared_from_this());
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    if (!reg_ctx)
        return false;
    
    Process *process = exe_ctx.GetProcessPtr();
    Error error;
    uint32_t fp_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP);
    uint32_t pc_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
    uint32_t sp_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber (eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
    
    // Do the argument layout
    
    std::vector <uint32_t> argLayout;   // 4-byte chunks, as discussed in the ABI Function Call Guide
    
    size_t numArgs = args.GetSize();
    size_t index;
    
    for (index = 0; index < numArgs; ++index)
    {
        Value *val = args.GetValueAtIndex(index);
        
        if (!val)
            return false;
        
        switch (val->GetValueType())
        {
        case Value::eValueTypeScalar:
            {
                Scalar &scalar = val->GetScalar();
                switch (scalar.GetType())
                {
                case Scalar::e_void:
                default:
                    return false;
                case Scalar::e_sint: 
                case Scalar::e_uint:
                case Scalar::e_slong:
                case Scalar::e_ulong:
                case Scalar::e_slonglong:
                case Scalar::e_ulonglong:
                    {
                        uint64_t data = scalar.ULongLong();
                        
                        switch (scalar.GetByteSize())
                        {
                        default:
                            return false;
                        case 1:
                            argLayout.push_back((uint32_t)(data & 0xffull));
                            break;
                        case 2:
                            argLayout.push_back((uint32_t)(data & 0xffffull));
                            break;
                        case 4:
                            argLayout.push_back((uint32_t)(data & 0xffffffffull));
                            break;
                        case 8:
                            argLayout.push_back((uint32_t)(data & 0xffffffffull));
                            argLayout.push_back((uint32_t)(data >> 32));
                            break;
                        }
                    }
                    break;
                case Scalar::e_float:
                    {
                        float data = scalar.Float();
                        uint32_t dataRaw = *((uint32_t*)(&data));
                        argLayout.push_back(dataRaw);
                    }
                    break;
                case Scalar::e_double:
                    {
                        double data = scalar.Double();
                        uint32_t *dataRaw = ((uint32_t*)(&data));
                        argLayout.push_back(dataRaw[0]);
                        argLayout.push_back(dataRaw[1]);
                    }
                    break;
                case Scalar::e_long_double:
                    {
                        long double data = scalar.Double();
                        uint32_t *dataRaw = ((uint32_t*)(&data));
                        while ((argLayout.size() * 4) & 0xf)
                            argLayout.push_back(0);
                        argLayout.push_back(dataRaw[0]);
                        argLayout.push_back(dataRaw[1]);
                        argLayout.push_back(dataRaw[2]);
                        argLayout.push_back(dataRaw[3]);
                    }
                    break;
                }
            }
            break;
        case Value::eValueTypeHostAddress:
            switch (val->GetContextType()) 
            {
            default:
                return false;
            case Value::eContextTypeClangType:
                {
                    void *val_type = val->GetClangType();
                    uint32_t cstr_length;
                    
                    if (ClangASTContext::IsCStringType (val_type, cstr_length))
                    {
                        const char *cstr = (const char*)val->GetScalar().ULongLong();
                        cstr_length = strlen(cstr);
                        
                        // Push the string onto the stack immediately.
                        
                        sp -= (cstr_length + 1);
                        
                        if (process->WriteMemory(sp, cstr, cstr_length + 1, error) != (cstr_length + 1))
                            return false;
                        
                        // Put the address of the string into the argument array.
                        
                        argLayout.push_back((uint32_t)(sp & 0xffffffff));
                    }
                    else
                    {
                        return false;
                    }
                }
                break;
            }
            break;
        case Value::eValueTypeFileAddress:
        case Value::eValueTypeLoadAddress:
        default:
            return false;
        }
    }
    
    // Make room for the arguments on the stack
    
    sp -= 4 * argLayout.size();
    
    // Align the SP
    
    sp &= ~(16ull-1ull); // 16-byte alignment
    
    // Write the arguments on the stack
    
    size_t numChunks = argLayout.size();

    for (index = 0; index < numChunks; ++index)
        if (process->WriteMemory(sp + (index * 4), &argLayout[index], sizeof(uint32_t), error) != sizeof(uint32_t))
            return false;
    
    // The return address is pushed onto the stack.
    
    sp -= 4;
    uint32_t returnAddressU32 = return_addr;
    if (process->WriteMemory (sp, &returnAddressU32, sizeof(returnAddressU32), error) != sizeof(returnAddressU32))
        return false;
    
    // %esp is set to the actual stack value.
    
    if (!reg_ctx->WriteRegisterFromUnsigned(sp_reg_num, sp))
        return false;
    
    // %ebp is set to a fake value, in our case 0x0x00000000
    
    if (!reg_ctx->WriteRegisterFromUnsigned(fp_reg_num, 0x00000000))
        return false;
    
    // %eip is set to the address of the called function.
    
    if (!reg_ctx->WriteRegisterFromUnsigned(pc_reg_num, func_addr))
        return false;
    
    return true;    
}

static bool 
ReadIntegerArgument (Scalar           &scalar,
                     unsigned int     bit_width,
                     bool             is_signed,
                     Process          *process,
                     addr_t           &current_stack_argument)
{
    
    uint32_t byte_size = (bit_width + (8-1))/8;
    Error error;
    if (process->ReadScalarIntegerFromMemory(current_stack_argument, byte_size, is_signed, scalar, error))
    {
        current_stack_argument += byte_size;
        return true;
    }
    return false;
}

bool
ABIMacOSX_i386::GetArgumentValues (Thread &thread,
                                   ValueList &values) const
{
    unsigned int num_values = values.GetSize();
    unsigned int value_index;
    
    // Extract the Clang AST context from the PC so that we can figure out type
    // sizes
    
    clang::ASTContext *ast_context = thread.CalculateTarget()->GetScratchClangASTContext()->getASTContext();
    
    // Get the pointer to the first stack argument so we have a place to start 
    // when reading data
    
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
    
    if (!reg_ctx)
        return false;
    
    addr_t sp = reg_ctx->GetSP(0);
    
    if (!sp)
        return false;
    
    addr_t current_stack_argument = sp + 4; // jump over return address
    
    for (value_index = 0;
         value_index < num_values;
         ++value_index)
    {
        Value *value = values.GetValueAtIndex(value_index);
        
        if (!value)
            return false;
        
        // We currently only support extracting values with Clang QualTypes.
        // Do we care about others?
        switch (value->GetContextType())
        {
            default:
                return false;
            case Value::eContextTypeClangType:
                {
                    void *value_type = value->GetClangType();
                    bool is_signed;
                    
                    if (ClangASTContext::IsIntegerType (value_type, is_signed))
                    {
                        size_t bit_width = ClangASTType::GetClangTypeBitWidth(ast_context, value_type);
                        
                        ReadIntegerArgument(value->GetScalar(),
                                            bit_width, 
                                            is_signed,
                                            thread.GetProcess().get(), 
                                            current_stack_argument);
                    }
                    else if (ClangASTContext::IsPointerType (value_type))
                    {
                        ReadIntegerArgument(value->GetScalar(),
                                            32,
                                            false,
                                            thread.GetProcess().get(),
                                            current_stack_argument);
                    }
                }
                break;
        }
    }
    
    return true;
}

Error
ABIMacOSX_i386::SetReturnValueObject(lldb::StackFrameSP &frame_sp, lldb::ValueObjectSP &new_value_sp)
{
    Error error;
    if (!new_value_sp)
    {
        error.SetErrorString("Empty value object for return value.");
        return error;
    }
    
    clang_type_t value_type = new_value_sp->GetClangType();
    if (!value_type)
    {
        error.SetErrorString ("Null clang type for return value.");
        return error;
    }
    
    clang::ASTContext *ast_context = new_value_sp->GetClangAST();
    if (!ast_context)
    {
        error.SetErrorString ("Null clang AST for return value.");
        return error;
    }
    Thread *thread = frame_sp->GetThread().get();
    
    bool is_signed;
    uint32_t count;
    bool is_complex;
    
    RegisterContext *reg_ctx = thread->GetRegisterContext().get();

    bool set_it_simple = false;
    if (ClangASTContext::IsIntegerType (value_type, is_signed) || ClangASTContext::IsPointerType(value_type))
    {
        DataExtractor data;
        size_t num_bytes = new_value_sp->GetData(data);
        uint32_t offset = 0;
        if (num_bytes <= 8)
        {
            const RegisterInfo *eax_info = reg_ctx->GetRegisterInfoByName("eax", 0);
            if (num_bytes <= 4)
            {
                uint32_t raw_value = data.GetMaxU32(&offset, num_bytes);
        
                if (reg_ctx->WriteRegisterFromUnsigned (eax_info, raw_value))
                    set_it_simple = true;
            }
            else
            {
                uint32_t raw_value = data.GetMaxU32(&offset, 4);
        
                if (reg_ctx->WriteRegisterFromUnsigned (eax_info, raw_value))
                {
                    const RegisterInfo *edx_info = reg_ctx->GetRegisterInfoByName("edx", 0);
                    uint32_t raw_value = data.GetMaxU32(&offset, num_bytes - offset);
                
                    if (reg_ctx->WriteRegisterFromUnsigned (edx_info, raw_value))
                        set_it_simple = true;
                }
            }
        }
        else
        {
            error.SetErrorString("We don't support returning longer than 64 bit integer values at present.");
        }
    }
    else if (ClangASTContext::IsFloatingPointType (value_type, count, is_complex))
    {
        if (is_complex)
            error.SetErrorString ("We don't support returning complex values at present");
        else
            error.SetErrorString ("We don't support returning float values at present");
    }
    
    if (!set_it_simple)
        error.SetErrorString ("We only support setting simple integer return types at present.");
    
    return error;
}

ValueObjectSP
ABIMacOSX_i386::GetReturnValueObjectImpl (Thread &thread,
                                ClangASTType &ast_type) const
{
    Value value;
    ValueObjectSP return_valobj_sp;
    
    void *value_type = ast_type.GetOpaqueQualType();
    if (!value_type) 
        return return_valobj_sp;
    
    clang::ASTContext *ast_context = ast_type.GetASTContext();
    if (!ast_context)
        return return_valobj_sp;

    value.SetContext (Value::eContextTypeClangType, value_type);
    
    RegisterContext *reg_ctx = thread.GetRegisterContext().get();
        if (!reg_ctx)
        return return_valobj_sp;
        
    bool is_signed;
            
    if (ClangASTContext::IsIntegerType (value_type, is_signed))
    {
        size_t bit_width = ClangASTType::GetClangTypeBitWidth(ast_context, value_type);
        
        unsigned eax_id = reg_ctx->GetRegisterInfoByName("eax", 0)->kinds[eRegisterKindLLDB];
        unsigned edx_id = reg_ctx->GetRegisterInfoByName("edx", 0)->kinds[eRegisterKindLLDB];
        
        switch (bit_width)
        {
            default:
            case 128:
                // Scalar can't hold 128-bit literals, so we don't handle this
                return return_valobj_sp;
            case 64:
                uint64_t raw_value;
                raw_value = thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff;
                raw_value |= (thread.GetRegisterContext()->ReadRegisterAsUnsigned(edx_id, 0) & 0xffffffff) << 32;
                if (is_signed)
                    value.GetScalar() = (int64_t)raw_value;
                else
                    value.GetScalar() = (uint64_t)raw_value;
                break;
            case 32:
                if (is_signed)
                    value.GetScalar() = (int32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff);
                else
                    value.GetScalar() = (uint32_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff);
                break;
            case 16:
                if (is_signed)
                    value.GetScalar() = (int16_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffff);
                else
                    value.GetScalar() = (uint16_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffff);
                break;
            case 8:
                if (is_signed)
                    value.GetScalar() = (int8_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xff);
                else
                    value.GetScalar() = (uint8_t)(thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xff);
                break;
        }
    }
    else if (ClangASTContext::IsPointerType (value_type))
    {
        unsigned eax_id = reg_ctx->GetRegisterInfoByName("eax", 0)->kinds[eRegisterKindLLDB];
        uint32_t ptr = thread.GetRegisterContext()->ReadRegisterAsUnsigned(eax_id, 0) & 0xffffffff;
        value.GetScalar() = ptr;
    }
    else
    {
        // not handled yet
        return return_valobj_sp;
    }
    
    // If we get here, we have a valid Value, so make our ValueObject out of it:
    
    return_valobj_sp = ValueObjectConstResult::Create(
                                    thread.GetStackFrameAtIndex(0).get(),
                                    ast_type.GetASTContext(),
                                    value,
                                    ConstString(""));
    return return_valobj_sp;
}

bool
ABIMacOSX_i386::CreateFunctionEntryUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t reg_kind = unwind_plan.GetRegisterKind();
    uint32_t sp_reg_num = LLDB_INVALID_REGNUM;
    uint32_t pc_reg_num = LLDB_INVALID_REGNUM;
    
    switch (reg_kind)
    {
        case eRegisterKindDWARF:
            sp_reg_num = dwarf_esp;
            pc_reg_num = dwarf_eip;
            break;

        case eRegisterKindGCC:
            sp_reg_num = gcc_esp;
            pc_reg_num = gcc_eip;
            break;
            
        case eRegisterKindGDB:
            sp_reg_num = gdb_esp;
            pc_reg_num = gdb_eip;
            break;
            
        case eRegisterKindGeneric:
            sp_reg_num = LLDB_REGNUM_GENERIC_SP;
            pc_reg_num = LLDB_REGNUM_GENERIC_PC;
            break;
    }
    
    if (sp_reg_num == LLDB_INVALID_REGNUM ||
        pc_reg_num == LLDB_INVALID_REGNUM)
        return false;

    UnwindPlan::RowSP row(new UnwindPlan::Row);
    row->SetCFARegister (sp_reg_num);
    row->SetCFAOffset (4);
    row->SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, -4, false);
    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("i386 at-func-entry default");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    return true;
}

bool
ABIMacOSX_i386::CreateDefaultUnwindPlan (UnwindPlan &unwind_plan)
{
    uint32_t fp_reg_num = dwarf_ebp;
    uint32_t sp_reg_num = dwarf_esp;
    uint32_t pc_reg_num = dwarf_eip;
    
    UnwindPlan::RowSP row(new UnwindPlan::Row);
    const int32_t ptr_size = 4;

    unwind_plan.Clear ();
    unwind_plan.SetRegisterKind (eRegisterKindDWARF);
    row->SetCFARegister (fp_reg_num);
    row->SetCFAOffset (2 * ptr_size);
    row->SetOffset (0);
    
    row->SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, ptr_size * -2, true);
    row->SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * -1, true);
    row->SetRegisterLocationToAtCFAPlusOffset(sp_reg_num, ptr_size *  0, true);

    unwind_plan.AppendRow (row);
    unwind_plan.SetSourceName ("i386 default unwind plan");
    unwind_plan.SetSourcedFromCompiler (eLazyBoolNo);
    unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    return true;
}

bool
ABIMacOSX_i386::RegisterIsVolatile (const RegisterInfo *reg_info)
{
    return RegisterIsCalleeSaved (reg_info);
}

bool
ABIMacOSX_i386::RegisterIsCalleeSaved (const RegisterInfo *reg_info)
{
    if (reg_info)
    {
        // Volatile registers include: ebx, ebp, esi, edi, esp, eip
        const char *name = reg_info->name;
        if (name[0] == 'e')
        {
            switch (name[1])
            {
            case 'b': 
                if (name[2] == 'x') // ebp is volatile in the ABI, but the unwinders can find it
                    return name[3] == '\0';
                break;
            case 'd':
                if (name[2] == 'i')
                    return name[3] == '\0';
                break;
            case 'i': 
                if (name[2] == 'p')
                    return name[3] == '\0';
                break;
            case 's':
                if (name[2] == 'i' || name[2] == 'p')
                    return name[3] == '\0';
                break;
            }
        }
    }
    return false;
}

void
ABIMacOSX_i386::Initialize()
{
    PluginManager::RegisterPlugin (pluginName,
                                   pluginDesc,
                                   CreateInstance);    
}

void
ABIMacOSX_i386::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ABIMacOSX_i386::GetPluginName()
{
    return pluginName;
}

const char *
ABIMacOSX_i386::GetShortPluginName()
{
    return pluginShort;
}

uint32_t
ABIMacOSX_i386::GetPluginVersion()
{
    return 1;
}

