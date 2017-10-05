//===-- RegisterInfos_ppc64le.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifdef DECLARE_REGISTER_INFOS_PPC64LE_STRUCT

// C Includes
#include <stddef.h>

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname) (offsetof(GPR, regname))
#define GPR_SIZE(regname) (sizeof(((GPR *)NULL)->regname))

#include "Utility/PPC64LE_DWARF_Registers.h"
#include "lldb-ppc64le-register-enums.h"

// Note that the size and offset will be updated by platform-specific classes.
#define DEFINE_GPR(reg, alt, lldb_kind)                                        \
  {                                                                            \
    #reg, alt, GPR_SIZE(reg), GPR_OFFSET(reg), lldb::eEncodingUint,            \
                                         lldb::eFormatHex,                     \
                                         {ppc64le_dwarf::dwarf_##reg##_ppc64le,\
                                          ppc64le_dwarf::dwarf_##reg##_ppc64le,\
                                          lldb_kind,                           \
                                          LLDB_INVALID_REGNUM,                 \
                                          gpr_##reg##_ppc64le },               \
                                          NULL, NULL, NULL, 0                  \
  }

// General purpose registers.
// EH_Frame, Generic, Process Plugin
#define POWERPC_REGS                                                           \
  DEFINE_GPR(r0, NULL, LLDB_INVALID_REGNUM)                                    \
  , DEFINE_GPR(r1, "sp", LLDB_REGNUM_GENERIC_SP),                              \
      DEFINE_GPR(r2, NULL, LLDB_INVALID_REGNUM),                               \
      DEFINE_GPR(r3, "arg1", LLDB_REGNUM_GENERIC_ARG1),                        \
      DEFINE_GPR(r4, "arg2", LLDB_REGNUM_GENERIC_ARG2),                        \
      DEFINE_GPR(r5, "arg3", LLDB_REGNUM_GENERIC_ARG3),                        \
      DEFINE_GPR(r6, "arg4", LLDB_REGNUM_GENERIC_ARG4),                        \
      DEFINE_GPR(r7, "arg5", LLDB_REGNUM_GENERIC_ARG5),                        \
      DEFINE_GPR(r8, "arg6", LLDB_REGNUM_GENERIC_ARG6),                        \
      DEFINE_GPR(r9, "arg7", LLDB_REGNUM_GENERIC_ARG7),                        \
      DEFINE_GPR(r10, "arg8", LLDB_REGNUM_GENERIC_ARG8),                       \
      DEFINE_GPR(r11, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r12, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r13, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r14, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r15, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r16, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r17, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r18, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r19, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r20, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r21, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r22, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r23, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r24, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r25, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r26, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r27, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r28, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r29, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r30, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(r31, NULL, LLDB_INVALID_REGNUM),                              \
      DEFINE_GPR(pc, "pc", LLDB_REGNUM_GENERIC_PC),                            \
      DEFINE_GPR(lr, "lr", LLDB_REGNUM_GENERIC_RA),                            \
      DEFINE_GPR(msr, "msr", LLDB_INVALID_REGNUM),                             \
      DEFINE_GPR(origr3, "orig_r3", LLDB_INVALID_REGNUM),                      \
      DEFINE_GPR(ctr, "ctr", LLDB_INVALID_REGNUM),                             \
      DEFINE_GPR(xer, "xer", LLDB_INVALID_REGNUM),                             \
      DEFINE_GPR(cr, "cr", LLDB_REGNUM_GENERIC_FLAGS),                         \
      DEFINE_GPR(softe, "softe", LLDB_INVALID_REGNUM),                         \
      DEFINE_GPR(trap, "trap", LLDB_INVALID_REGNUM),                           \
      /* */

typedef struct _GPR {
  uint64_t r0;
  uint64_t r1;
  uint64_t r2;
  uint64_t r3;
  uint64_t r4;
  uint64_t r5;
  uint64_t r6;
  uint64_t r7;
  uint64_t r8;
  uint64_t r9;
  uint64_t r10;
  uint64_t r11;
  uint64_t r12;
  uint64_t r13;
  uint64_t r14;
  uint64_t r15;
  uint64_t r16;
  uint64_t r17;
  uint64_t r18;
  uint64_t r19;
  uint64_t r20;
  uint64_t r21;
  uint64_t r22;
  uint64_t r23;
  uint64_t r24;
  uint64_t r25;
  uint64_t r26;
  uint64_t r27;
  uint64_t r28;
  uint64_t r29;
  uint64_t r30;
  uint64_t r31;
  uint64_t pc;
  uint64_t msr;
  uint64_t origr3;
  uint64_t ctr;
  uint64_t lr;
  uint64_t xer;
  uint64_t cr;
  uint64_t softe;
  uint64_t trap;
  uint64_t pad[4];
} GPR;

static lldb_private::RegisterInfo g_register_infos_ppc64le[] = {
    POWERPC_REGS
};

static_assert((sizeof(g_register_infos_ppc64le) /
               sizeof(g_register_infos_ppc64le[0])) ==
                  k_num_registers_ppc64le,
              "g_register_infos_powerpc64 has wrong number of register infos");

#undef DEFINE_GPR

#endif // DECLARE_REGISTER_INFOS_PPC64LE_STRUCT
