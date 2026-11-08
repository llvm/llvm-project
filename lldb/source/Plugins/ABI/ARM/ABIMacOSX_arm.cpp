//===-- ABIMacOSX_arm.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIMacOSX_arm.h"
#include "lldb/Register/Arch/lldb-arm-ehframe-register-enums.h"

#include <optional>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/TargetParser/Triple.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ThreadRegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"

#include "Plugins/Process/Utility/ARMDefines.h"
#include "lldb/Register/Arch/lldb-arm-dwarf-register-enums.h"

namespace dwarf_regs = lldb_private::registers::arm::dwarf;
namespace ehframe_regs = lldb_private::registers::arm::ehframe;
using namespace lldb;
using namespace lldb_private;

static const RegisterDescriptor g_register_infos[] = {
    RegisterDescriptor(536870912, "r0", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r0, dwarf_regs::r0, LLDB_REGNUM_GENERIC_ARG1,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870913, "r1", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r1, dwarf_regs::r1, LLDB_REGNUM_GENERIC_ARG2,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870914, "r2", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r2, dwarf_regs::r2, LLDB_REGNUM_GENERIC_ARG3,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870915, "r3", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r3, dwarf_regs::r3, LLDB_REGNUM_GENERIC_ARG4,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870916, "r4", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r4, dwarf_regs::r4, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870917, "r5", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r5, dwarf_regs::r5, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870918, "r6", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r6, dwarf_regs::r6, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870919, "r7", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r7, dwarf_regs::r7, LLDB_REGNUM_GENERIC_FP,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870920, "r8", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r8, dwarf_regs::r8, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870921, "r9", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r9, dwarf_regs::r9, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870922, "r10", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r10, dwarf_regs::r10, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870923, "r11", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r11, dwarf_regs::r11, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870924, "r12", nullptr, 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::r12, dwarf_regs::r12, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870925, "sp", "r13", 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::sp, dwarf_regs::sp, LLDB_REGNUM_GENERIC_SP,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870926, "lr", "r14", 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::lr, dwarf_regs::lr, LLDB_REGNUM_GENERIC_RA,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870927, "pc", "r15", 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::pc, dwarf_regs::pc, LLDB_REGNUM_GENERIC_PC,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870928, "cpsr", "psr", 4, eEncodingUint, eFormatHex,
                       {ehframe_regs::cpsr, dwarf_regs::cpsr, LLDB_REGNUM_GENERIC_FLAGS,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870929, "s0", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s0, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870930, "s1", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s1, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870931, "s2", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s2, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870932, "s3", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s3, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870933, "s4", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s4, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870934, "s5", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s5, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870935, "s6", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s6, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870936, "s7", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s7, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870937, "s8", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s8, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870938, "s9", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s9, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870939, "s10", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s10, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870940, "s11", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s11, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870941, "s12", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s12, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870942, "s13", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s13, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870943, "s14", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s14, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870944, "s15", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s15, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870945, "s16", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s16, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870946, "s17", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s17, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870947, "s18", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s18, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870948, "s19", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s19, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870949, "s20", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s20, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870950, "s21", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s21, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870951, "s22", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s22, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870952, "s23", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s23, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870953, "s24", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s24, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870954, "s25", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s25, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870955, "s26", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s26, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870956, "s27", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s27, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870957, "s28", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s28, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870958, "s29", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s29, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870959, "s30", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s30, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870960, "s31", nullptr, 4, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::s31, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(
        536870961, "fpscr", nullptr, 4, eEncodingUint, eFormatHex,
        {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
         LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
        {}, {}, nullptr),
    RegisterDescriptor(536870962, "d0", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d0, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870963, "d1", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d1, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870964, "d2", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d2, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870965, "d3", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d3, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870966, "d4", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d4, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870967, "d5", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d5, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870968, "d6", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d6, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870969, "d7", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d7, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870970, "d8", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d8, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870971, "d9", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d9, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870972, "d10", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d10, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870973, "d11", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d11, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870974, "d12", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d12, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870975, "d13", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d13, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870976, "d14", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d14, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870977, "d15", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d15, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870978, "d16", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d16, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870979, "d17", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d17, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870980, "d18", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d18, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870981, "d19", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d19, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870982, "d20", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d20, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870983, "d21", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d21, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870984, "d22", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d22, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870985, "d23", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d23, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870986, "d24", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d24, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870987, "d25", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d25, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870988, "d26", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d26, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870989, "d27", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d27, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870990, "d28", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d28, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870991, "d29", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d29, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870992, "d30", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d30, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870993, "d31", nullptr, 8, eEncodingIEEE754,
                       eFormatFloat,
                       {LLDB_INVALID_REGNUM, dwarf_regs::d31, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870994, "r8_usr", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r8_usr, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870995, "r9_usr", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r9_usr, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870996, "r10_usr", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r10_usr, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870997, "r11_usr", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r11_usr, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870998, "r12_usr", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r12_usr, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536870999, "r13_usr", "sp_usr", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r13_usr, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871000, "r14_usr", "lr_usr", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r14_usr, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871001, "r8_fiq", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r8_fiq, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871002, "r9_fiq", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r9_fiq, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871003, "r10_fiq", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r10_fiq, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871004, "r11_fiq", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r11_fiq, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871005, "r12_fiq", nullptr, 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r12_fiq, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871006, "r13_fiq", "sp_fiq", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r13_fiq, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871007, "r14_fiq", "lr_fiq", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r14_fiq, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871008, "r13_irq", "sp_irq", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r13_irq, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871009, "r14_irq", "lr_irq", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r14_irq, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871010, "r13_abt", "sp_abt", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r13_abt, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871011, "r14_abt", "lr_abt", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r14_abt, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871012, "r13_und", "sp_und", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r13_und, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871013, "r14_und", "lr_und", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r14_und, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871014, "r13_svc", "sp_svc", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r13_svc, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr),
    RegisterDescriptor(536871015, "r14_svc", "lr_svc", 4, eEncodingUint,
                       eFormatHex,
                       {LLDB_INVALID_REGNUM, dwarf_regs::r14_svc, LLDB_INVALID_REGNUM,
                        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM},
                       {}, {}, nullptr)};

static const uint32_t k_num_register_infos = std::size(g_register_infos);

const lldb_private::RegisterDescriptor *
ABIMacOSX_arm::GetRegisterInfoArray(uint32_t &count) {
  count = k_num_register_infos;
  return g_register_infos;
}

size_t ABIMacOSX_arm::GetRedZoneSize() const { return 0; }

// Static Functions

ABISP
ABIMacOSX_arm::CreateInstance(ProcessSP process_sp, const ArchSpec &arch) {
  const llvm::Triple::ArchType arch_type = arch.GetTriple().getArch();
  const llvm::Triple::VendorType vendor_type = arch.GetTriple().getVendor();

  if (vendor_type == llvm::Triple::Apple) {
    if ((arch_type == llvm::Triple::arm) ||
        (arch_type == llvm::Triple::thumb)) {
      return ABISP(
          new ABIMacOSX_arm(std::move(process_sp), MakeMCRegisterInfo(arch)));
    }
  }

  return ABISP();
}

bool ABIMacOSX_arm::PrepareTrivialCall(Thread &thread, addr_t sp,
                                       addr_t function_addr, addr_t return_addr,
                                       llvm::ArrayRef<addr_t> args) const {
  ThreadRegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return false;

  const uint32_t pc_reg_num = reg_ctx->GetRegisterIDForNumber(
      eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  const uint32_t sp_reg_num = reg_ctx->GetRegisterIDForNumber(
      eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
  const uint32_t ra_reg_num = reg_ctx->GetRegisterIDForNumber(
      eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA);

  RegisterValue reg_value;

  const char *reg_names[] = {"r0", "r1", "r2", "r3"};

  llvm::ArrayRef<addr_t>::iterator ai = args.begin(), ae = args.end();

  for (size_t i = 0; i < std::size(reg_names); ++i) {
    if (ai == ae)
      break;

    reg_value.SetUInt32(*ai);
    if (llvm::errorToBool(reg_ctx->WriteRegister(
            (reg_ctx->GetRegisterByName(reg_names[i])
                 ? (reg_ctx->GetRegisterByName(reg_names[i]))->id
                 : LLDB_INVALID_REGNUM),
            reg_value)))
      return false;

    ++ai;
  }

  if (ai != ae) {
    // Spill onto the stack
    size_t num_stack_regs = ae - ai;

    sp -= (num_stack_regs * 4);
    // Keep the stack 16 byte aligned
    sp &= ~(16ull - 1ull);

    // just using arg1 to get the right size
    const RegisterDescriptor *reg_info = reg_ctx->GetRegisterByNumber(
        eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1);

    addr_t arg_pos = sp;

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

  TargetSP target_sp(thread.CalculateTarget());
  Address so_addr;

  // Figure out if our return address is ARM or Thumb by using the
  // Address::GetCallableLoadAddress(Target*) which will figure out the ARM
  // thumb-ness and set the correct address bits for us.
  so_addr.SetLoadAddress(return_addr, target_sp.get());
  return_addr = so_addr.GetCallableLoadAddress(target_sp.get());

  // Set "lr" to the return address
  if (llvm::errorToBool(
          reg_ctx->WriteRegisterFromUnsigned(ra_reg_num, return_addr)))
    return false;

  // If bit zero or 1 is set, this must be a thumb function, no need to figure
  // this out from the symbols.
  so_addr.SetLoadAddress(function_addr, target_sp.get());
  function_addr = so_addr.GetCallableLoadAddress(target_sp.get());

  const RegisterDescriptor *cpsr_reg_info =
      reg_ctx->GetRegisterByName("cpsr");
  const uint32_t curr_cpsr = reg_ctx->ReadRegisterAsUnsigned(
      (cpsr_reg_info ? cpsr_reg_info->id : LLDB_INVALID_REGNUM), 0);

  // Make a new CPSR and mask out any Thumb IT (if/then) bits
  uint32_t new_cpsr = curr_cpsr & ~MASK_CPSR_IT_MASK;
  // If bit zero or 1 is set, this must be thumb...
  if (function_addr & 1ull)
    new_cpsr |= MASK_CPSR_T; // Set T bit in CPSR
  else
    new_cpsr &= ~MASK_CPSR_T; // Clear T bit in CPSR

  if (new_cpsr != curr_cpsr) {
    if (llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
            (cpsr_reg_info ? cpsr_reg_info->id : LLDB_INVALID_REGNUM),
            new_cpsr)))
      return false;
  }

  function_addr &=
      ~1ull; // clear bit zero since the CPSR will take care of the mode for us

  // Update the sp - stack pointer - to be aligned to 16-bytes
  sp &= ~(0xfull);
  if (llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(sp_reg_num, sp)))
    return false;

  // Set "pc" to the address requested
  if (llvm::errorToBool(
          reg_ctx->WriteRegisterFromUnsigned(pc_reg_num, function_addr)))
    return false;

  return true;
}

bool ABIMacOSX_arm::GetArgumentValues(Thread &thread, ValueList &values) const {
  uint32_t num_values = values.GetSize();

  ExecutionContext exe_ctx(thread.shared_from_this());
  // For now, assume that the types in the AST values come from the Target's
  // scratch AST.

  // Extract the register context so we can read arguments from registers

  ThreadRegisterContext *reg_ctx = thread.GetRegisterContext().get();

  if (!reg_ctx)
    return false;

  addr_t sp = 0;

  for (uint32_t value_idx = 0; value_idx < num_values; ++value_idx) {
    // We currently only support extracting values with Clang QualTypes. Do we
    // care about others?
    Value *value = values.GetValueAtIndex(value_idx);

    if (!value)
      return false;

    CompilerType compiler_type = value->GetCompilerType();
    if (compiler_type) {
      bool is_signed = false;
      size_t bit_width = 0;
      std::optional<uint64_t> bit_size =
          llvm::expectedToOptional(compiler_type.GetBitSize(&thread));
      if (!bit_size)
        return false;
      if (compiler_type.IsIntegerOrEnumerationType(is_signed))
        bit_width = *bit_size;
      else if (compiler_type.IsPointerOrReferenceType())
        bit_width = *bit_size;
      else
        // We only handle integer, pointer and reference types currently...
        return false;

      if (bit_width <= (exe_ctx.GetProcessRef().GetAddressByteSize() * 8)) {
        if (value_idx < 4) {
          // Arguments 1-4 are in r0-r3...
          const RegisterDescriptor *arg_reg_info = nullptr;
          // Search by generic ID first, then fall back to by name
          uint32_t arg_reg_num = reg_ctx->GetRegisterIDForNumber(
              eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1 + value_idx);
          if (arg_reg_num != LLDB_INVALID_REGNUM) {
            arg_reg_info = reg_ctx->GetDescription()->Find(arg_reg_num);
          } else {
            switch (value_idx) {
            case 0:
              arg_reg_info = reg_ctx->GetRegisterByName("r0");
              break;
            case 1:
              arg_reg_info = reg_ctx->GetRegisterByName("r1");
              break;
            case 2:
              arg_reg_info = reg_ctx->GetRegisterByName("r2");
              break;
            case 3:
              arg_reg_info = reg_ctx->GetRegisterByName("r3");
              break;
            }
          }

          if (arg_reg_info) {
            RegisterValue reg_value;

            if (!llvm::errorToBool(
                    reg_ctx
                        ->ReadRegister((arg_reg_info ? arg_reg_info->id
                                                     : LLDB_INVALID_REGNUM))
                        .moveInto(reg_value))) {
              if (is_signed)
                reg_value.SignExtend(bit_width);
              if (!reg_value.GetScalarValue(value->GetScalar()))
                return false;
              continue;
            }
          }
          return false;
        } else {
          if (sp == 0) {
            // Read the stack pointer if it already hasn't been read
            sp = reg_ctx->GetSP(0);
            if (sp == 0)
              return false;
          }

          // Arguments 5 on up are on the stack
          const uint32_t arg_byte_size = (bit_width + (8 - 1)) / 8;
          Status error;
          if (!exe_ctx.GetProcessRef().ReadScalarIntegerFromMemory(
                  sp, arg_byte_size, is_signed, value->GetScalar(), error))
            return false;

          sp += arg_byte_size;
        }
      }
    }
  }
  return true;
}

bool ABIMacOSX_arm::IsArmv7kProcess() const {
  bool is_armv7k = false;
  ProcessSP process_sp(GetProcessSP());
  if (process_sp) {
    const ArchSpec &arch(process_sp->GetTarget().GetArchitecture());
    const ArchSpec::Core system_core = arch.GetCore();
    if (system_core == ArchSpec::eCore_arm_armv7k) {
      is_armv7k = true;
    }
  }
  return is_armv7k;
}

ValueObjectSP ABIMacOSX_arm::GetReturnValueObjectImpl(
    Thread &thread, lldb_private::CompilerType &compiler_type) const {
  Value value;
  ValueObjectSP return_valobj_sp;

  if (!compiler_type)
    return return_valobj_sp;

  value.SetCompilerType(compiler_type);

  ThreadRegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return return_valobj_sp;

  bool is_signed;

  // Get the pointer to the first stack argument so we have a place to start
  // when reading data

  const RegisterDescriptor *r0_reg_info =
      reg_ctx->GetRegisterByName("r0", 0);
  if (compiler_type.IsIntegerOrEnumerationType(is_signed)) {
    std::optional<uint64_t> bit_width =
        llvm::expectedToOptional(compiler_type.GetBitSize(&thread));
    if (!bit_width)
      return return_valobj_sp;

    switch (*bit_width) {
    default:
      return return_valobj_sp;
    case 128:
      if (IsArmv7kProcess()) {
        // "A composite type not larger than 16 bytes is returned in r0-r3. The
        // format is as if the result had been stored in memory at a word-
        // aligned address and then loaded into r0-r3 with an ldm instruction"
        {
          const RegisterDescriptor *r1_reg_info =
              reg_ctx->GetRegisterByName("r1", 0);
          const RegisterDescriptor *r2_reg_info =
              reg_ctx->GetRegisterByName("r2", 0);
          const RegisterDescriptor *r3_reg_info =
              reg_ctx->GetRegisterByName("r3", 0);
          if (r1_reg_info && r2_reg_info && r3_reg_info) {
            std::optional<uint64_t> byte_size =
                llvm::expectedToOptional(compiler_type.GetByteSize(&thread));
            if (!byte_size)
              return return_valobj_sp;
            ProcessSP process_sp(thread.GetProcess());
            if (*byte_size <= r0_reg_info->byte_size + r1_reg_info->byte_size +
                                  r2_reg_info->byte_size +
                                  r3_reg_info->byte_size &&
                process_sp) {
              std::unique_ptr<DataBufferHeap> heap_data_up(
                  new DataBufferHeap(*byte_size, 0));
              const ByteOrder byte_order = process_sp->GetByteOrder();
              RegisterValue r0_reg_value;
              RegisterValue r1_reg_value;
              RegisterValue r2_reg_value;
              RegisterValue r3_reg_value;
              if (!llvm::errorToBool(
                      reg_ctx
                          ->ReadRegister((r0_reg_info ? r0_reg_info->id
                                                      : LLDB_INVALID_REGNUM))
                          .moveInto(r0_reg_value)) &&
                  !llvm::errorToBool(
                      reg_ctx
                          ->ReadRegister((r1_reg_info ? r1_reg_info->id
                                                      : LLDB_INVALID_REGNUM))
                          .moveInto(r1_reg_value)) &&
                  !llvm::errorToBool(
                      reg_ctx
                          ->ReadRegister((r2_reg_info ? r2_reg_info->id
                                                      : LLDB_INVALID_REGNUM))
                          .moveInto(r2_reg_value)) &&
                  !llvm::errorToBool(
                      reg_ctx
                          ->ReadRegister((r3_reg_info ? r3_reg_info->id
                                                      : LLDB_INVALID_REGNUM))
                          .moveInto(r3_reg_value))) {
                Status error;
                if (r0_reg_value.GetAsMemoryData(
                        r0_reg_info->byte_size, heap_data_up->GetBytes() + 0, 4,
                        byte_order, error, r0_reg_info->name) &&
                    r1_reg_value.GetAsMemoryData(
                        r1_reg_info->byte_size, heap_data_up->GetBytes() + 4, 4,
                        byte_order, error, r1_reg_info->name) &&
                    r2_reg_value.GetAsMemoryData(
                        r2_reg_info->byte_size, heap_data_up->GetBytes() + 8, 4,
                        byte_order, error, r2_reg_info->name) &&
                    r3_reg_value.GetAsMemoryData(
                        r3_reg_info->byte_size, heap_data_up->GetBytes() + 12,
                        4, byte_order, error, r3_reg_info->name)) {
                  DataExtractor data(DataBufferSP(heap_data_up.release()),
                                     byte_order,
                                     process_sp->GetAddressByteSize());

                  return_valobj_sp = ValueObjectConstResult::Create(
                      &thread, compiler_type, ConstString(""), data);
                  return return_valobj_sp;
                }
              }
            }
          }
        }
      } else {
        return return_valobj_sp;
      }
      break;
    case 64: {
      const RegisterDescriptor *r1_reg_info =
          reg_ctx->GetRegisterByName("r1", 0);
      uint64_t raw_value;
      raw_value =
          reg_ctx->ReadRegisterAsUnsigned(
              (r0_reg_info ? r0_reg_info->id : LLDB_INVALID_REGNUM), 0) &
          UINT32_MAX;
      raw_value |=
          ((uint64_t)(reg_ctx->ReadRegisterAsUnsigned(
                          (r1_reg_info ? r1_reg_info->id : LLDB_INVALID_REGNUM),
                          0) &
                               UINT32_MAX))
                   << 32;
      if (is_signed)
        value.GetScalar() = (int64_t)raw_value;
      else
        value.GetScalar() = (uint64_t)raw_value;
    } break;
    case 32:
      if (is_signed)
        value.GetScalar() =
            (int32_t)(reg_ctx->ReadRegisterAsUnsigned(
                          (r0_reg_info ? r0_reg_info->id : LLDB_INVALID_REGNUM),
                          0) &
                      UINT32_MAX);
      else
        value.GetScalar() = (uint32_t)(reg_ctx->ReadRegisterAsUnsigned(
                                           (r0_reg_info ? r0_reg_info->id
                                                        : LLDB_INVALID_REGNUM),
                                           0) &
                                       UINT32_MAX);
      break;
    case 16:
      if (is_signed)
        value.GetScalar() =
            (int16_t)(reg_ctx->ReadRegisterAsUnsigned(
                          (r0_reg_info ? r0_reg_info->id : LLDB_INVALID_REGNUM),
                          0) &
                      UINT16_MAX);
      else
        value.GetScalar() = (uint16_t)(reg_ctx->ReadRegisterAsUnsigned(
                                           (r0_reg_info ? r0_reg_info->id
                                                        : LLDB_INVALID_REGNUM),
                                           0) &
                                       UINT16_MAX);
      break;
    case 8:
      if (is_signed)
        value.GetScalar() =
            (int8_t)(reg_ctx->ReadRegisterAsUnsigned(
                         (r0_reg_info ? r0_reg_info->id : LLDB_INVALID_REGNUM),
                         0) &
                     UINT8_MAX);
      else
        value.GetScalar() =
            (uint8_t)(reg_ctx->ReadRegisterAsUnsigned(
                          (r0_reg_info ? r0_reg_info->id : LLDB_INVALID_REGNUM),
                          0) &
                      UINT8_MAX);
      break;
    }
  } else if (compiler_type.IsPointerType()) {
    uint32_t ptr =
        thread.GetRegisterContext()->ReadRegisterAsUnsigned(
            (r0_reg_info ? r0_reg_info->id : LLDB_INVALID_REGNUM), 0) &
        UINT32_MAX;
    value.GetScalar() = ptr;
  } else {
    // not handled yet
    return return_valobj_sp;
  }

  // If we get here, we have a valid Value, so make our ValueObject out of it:

  return_valobj_sp = ValueObjectConstResult::Create(
      thread.GetStackFrameAtIndex(0).get(), value, ConstString(""));
  return return_valobj_sp;
}

Status ABIMacOSX_arm::SetReturnValueObject(lldb::StackFrameSP &frame_sp,
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
      const RegisterDescriptor *r0_info =
          reg_ctx->GetRegisterByName("r0", 0);
      if (num_bytes <= 4) {
        uint32_t raw_value = data.GetMaxU32(&offset, num_bytes);

        if (!llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
                (r0_info ? r0_info->id : LLDB_INVALID_REGNUM), raw_value)))
          set_it_simple = true;
      } else {
        uint32_t raw_value = data.GetMaxU32(&offset, 4);

        if (!llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
                (r0_info ? r0_info->id : LLDB_INVALID_REGNUM), raw_value))) {
          const RegisterDescriptor *r1_info =
              reg_ctx->GetRegisterByName("r1", 0);
          uint32_t raw_value = data.GetMaxU32(&offset, num_bytes - offset);

          if (!llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
                  (r1_info ? r1_info->id : LLDB_INVALID_REGNUM), raw_value)))
            set_it_simple = true;
        }
      }
    } else if (num_bytes <= 16 && IsArmv7kProcess()) {
      // "A composite type not larger than 16 bytes is returned in r0-r3. The
      // format is as if the result had been stored in memory at a word-aligned
      // address and then loaded into r0-r3 with an ldm instruction"

      const RegisterDescriptor *r0_info =
          reg_ctx->GetRegisterByName("r0", 0);
      const RegisterDescriptor *r1_info =
          reg_ctx->GetRegisterByName("r1", 0);
      const RegisterDescriptor *r2_info =
          reg_ctx->GetRegisterByName("r2", 0);
      const RegisterDescriptor *r3_info =
          reg_ctx->GetRegisterByName("r3", 0);
      lldb::offset_t offset = 0;
      uint32_t bytes_written = 4;
      uint32_t raw_value = data.GetMaxU64(&offset, 4);
      if (!llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
              (r0_info ? r0_info->id : LLDB_INVALID_REGNUM), raw_value)) &&
          bytes_written <= num_bytes) {
        bytes_written += 4;
        raw_value = data.GetMaxU64(&offset, 4);
        if (bytes_written <= num_bytes &&
            !llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
                (r1_info ? r1_info->id : LLDB_INVALID_REGNUM), raw_value))) {
          bytes_written += 4;
          raw_value = data.GetMaxU64(&offset, 4);
          if (bytes_written <= num_bytes &&
              !llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
                  (r2_info ? r2_info->id : LLDB_INVALID_REGNUM), raw_value))) {
            bytes_written += 4;
            raw_value = data.GetMaxU64(&offset, 4);
            if (bytes_written <= num_bytes &&
                !llvm::errorToBool(reg_ctx->WriteRegisterFromUnsigned(
                    (r3_info ? r3_info->id : LLDB_INVALID_REGNUM),
                    raw_value))) {
              set_it_simple = true;
            }
          }
        }
      }
    } else {
      error = Status::FromErrorString(
          "We don't support returning longer than 64 bit "
          "integer values at present.");
    }
  }

  if (!set_it_simple)
    error = Status::FromErrorString(
        "We only support setting simple integer return types at present.");

  return error;
}

UnwindPlanSP ABIMacOSX_arm::CreateFunctionEntryUnwindPlan() {
  uint32_t lr_reg_num = dwarf_regs::lr;
  uint32_t sp_reg_num = dwarf_regs::sp;
  uint32_t pc_reg_num = dwarf_regs::pc;

  UnwindPlan::Row row;

  // Our Call Frame Address is the stack pointer value
  row.GetCFAValue().SetIsRegisterPlusOffset(sp_reg_num, 0);

  // The previous PC is in the LR, all other registers are the same.
  row.SetRegisterLocationToRegister(pc_reg_num, lr_reg_num, true);

  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindDWARF);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("arm at-func-entry default");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  return plan_sp;
}

UnwindPlanSP ABIMacOSX_arm::CreateDefaultUnwindPlan() {
  uint32_t fp_reg_num =
      dwarf_regs::r7; // apple uses r7 for all frames. Normal arm uses r11
  uint32_t pc_reg_num = dwarf_regs::pc;

  UnwindPlan::Row row;
  const int32_t ptr_size = 4;

  row.GetCFAValue().SetIsRegisterPlusOffset(fp_reg_num, 2 * ptr_size);
  row.SetUnspecifiedRegistersAreUndefined(true);

  row.SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, ptr_size * -2, true);
  row.SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * -1, true);

  auto plan_sp = std::make_shared<UnwindPlan>(eRegisterKindDWARF);
  plan_sp->AppendRow(std::move(row));
  plan_sp->SetSourceName("arm-apple-ios default unwind plan");
  plan_sp->SetSourcedFromCompiler(eLazyBoolNo);
  plan_sp->SetUnwindPlanValidAtAllInstructions(eLazyBoolNo);
  plan_sp->SetUnwindPlanForSignalTrap(eLazyBoolNo);
  return plan_sp;
}

// cf. "ARMv6 Function Calling Conventions"
// https://developer.apple.com/library/ios/documentation/Xcode/Conceptual/iPhoneOSABIReference/Articles/ARMv6FunctionCallingConventions.html
// and "ARMv7 Function Calling Conventions"
// https://developer.apple.com/library/ios/documentation/Xcode/Conceptual/iPhoneOSABIReference/Articles/ARMv7FunctionCallingConventions.html

// ARMv7 on iOS general purpose reg rules:
//    r0-r3 not preserved  (used for argument passing)
//    r4-r6 preserved
//    r7    preserved (frame pointer)
//    r8    preserved
//    r9    not preserved (usable as volatile scratch register with iOS 3.x and
//    later)
//    r10-r11 preserved
//    r12   not presrved
//    r13   preserved (stack pointer)
//    r14   not preserved (link register)
//    r15   preserved (pc)
//    cpsr  not preserved (different rules for different bits)

// ARMv7 on iOS floating point rules:
//    d0-d7   not preserved   (aka s0-s15, q0-q3)
//    d8-d15  preserved       (aka s16-s31, q4-q7)
//    d16-d31 not preserved   (aka q8-q15)

bool ABIMacOSX_arm::RegisterIsVolatile(const RegisterDescriptor *reg_info) {
  if (reg_info) {
    // Volatile registers are: r0, r1, r2, r3, r9, r12, r13 (aka sp)
    const char *name = reg_info->name.c_str();
    if (name[0] == 'r') {
      switch (name[1]) {
      case '0':
        return name[2] == '\0'; // r0
      case '1':
        switch (name[2]) {
        case '\0':
          return true; // r1
        case '2':
        case '3':
          return name[3] == '\0'; // r12, r13 (sp)
        default:
          break;
        }
        break;

      case '2':
        return name[2] == '\0'; // r2
      case '3':
        return name[2] == '\0'; // r3
      case '9':
        return name[2] == '\0'; // r9 (apple-ios only...)

        break;
      }
    } else if (name[0] == 'd') {
      switch (name[1]) {
      case '0':
        return name[2] == '\0'; // d0 is volatile

      case '1':
        switch (name[2]) {
        case '\0':
          return true; // d1 is volatile
        case '6':
        case '7':
        case '8':
        case '9':
          return name[3] == '\0'; // d16 - d19 are volatile
        default:
          break;
        }
        break;

      case '2':
        switch (name[2]) {
        case '\0':
          return true; // d2 is volatile
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
          return name[3] == '\0'; // d20 - d29 are volatile
        default:
          break;
        }
        break;

      case '3':
        switch (name[2]) {
        case '\0':
          return true; // d3 is volatile
        case '0':
        case '1':
          return name[3] == '\0'; // d30 - d31 are volatile
        default:
          break;
        }
        break;
      case '4':
      case '5':
      case '6':
      case '7':
        return name[2] == '\0'; // d4 - d7 are volatile

      default:
        break;
      }
    } else if (name[0] == 's') {
      switch (name[1]) {
      case '0':
        return name[2] == '\0'; // s0 is volatile

      case '1':
        switch (name[2]) {
        case '\0':
          return true; // s1 is volatile
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
          return name[3] == '\0'; // s10 - s15 are volatile
        default:
          break;
        }
        break;

      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        return name[2] == '\0'; // s2 - s9 are volatile

      default:
        break;
      }
    } else if (name[0] == 'q') {
      switch (name[1]) {
      case '1':
        switch (name[2]) {
        case '\0':
          return true; // q1 is volatile
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
          return true; // q10-q15 are volatile
        default:
          break;
        };
        break;
      case '0':
      case '2':
      case '3':
        return name[2] == '\0'; // q0-q3 are volatile
      case '8':
      case '9':
        return name[2] == '\0'; // q8-q9 are volatile
      default:
        break;
      }
    } else if (name[0] == 's' && name[1] == 'p' && name[2] == '\0')
      return true;
  }
  return false;
}

void ABIMacOSX_arm::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "Mac OS X ABI for arm targets", CreateInstance);
}

void ABIMacOSX_arm::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
