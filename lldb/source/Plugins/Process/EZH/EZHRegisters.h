//===-- EZHRegisters.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_EZHREGISTERS_H
#define LLDB_SYMBOL_EZHREGISTERS_H

// NXP SmartDMA (EZH) Peripheral Register Offsets
#define EZHB_BOOT_OFFSET        0x20
#define EZHB_CTRL_OFFSET        0x24
#define EZHB_PC_OFFSET          0x28
#define EZHB_SP_OFFSET          0x2C
#define EZHB_BREAK_ADDR_OFFSET  0x30
#define EZHB_BREAK_VECT_OFFSET  0x34
#define EZHB_EMER_VECT_OFFSET   0x38
#define EZHB_EMER_SEL_OFFSET    0x3C
#define EZHB_ARM2EZH_OFFSET     0x40
#define EZHB_EZH2ARM_OFFSET     0x44
#define EZHB_PENDTRAP_OFFSET    0x48

// EZH 68-byte Stack Frame Register Layout
#define EZH_FRAME_SIZE          68

#define EZH_FRAME_OFFSET_R0     (-68)
#define EZH_FRAME_OFFSET_R1     (-64)
#define EZH_FRAME_OFFSET_R2     (-60)
#define EZH_FRAME_OFFSET_R3     (-56)
#define EZH_FRAME_OFFSET_R4     (-52)
#define EZH_FRAME_OFFSET_R5     (-48)
#define EZH_FRAME_OFFSET_R6     (-44)
#define EZH_FRAME_OFFSET_R7     (-40)
#define EZH_FRAME_OFFSET_GPO    (-36)
#define EZH_FRAME_OFFSET_GPD    (-32)
#define EZH_FRAME_OFFSET_CFS    (-28)
#define EZH_FRAME_OFFSET_CFM    (-24)
#define EZH_FRAME_OFFSET_SP     (-20)
#define EZH_FRAME_OFFSET_PC     (-16)
#define EZH_FRAME_OFFSET_GPI    (-12)
#define EZH_FRAME_OFFSET_RA     (-8)
#define EZH_FRAME_OFFSET_FLAGS  (-4)

// LLDB register index defines
#define EZH_REG_IDX_R0          0
#define EZH_REG_IDX_R1          1
#define EZH_REG_IDX_R2          2
#define EZH_REG_IDX_R3          3
#define EZH_REG_IDX_R4          4
#define EZH_REG_IDX_R5          5
#define EZH_REG_IDX_R6          6
#define EZH_REG_IDX_R7          7
#define EZH_REG_IDX_GPO         8
#define EZH_REG_IDX_GPD         9
#define EZH_REG_IDX_CFS         10
#define EZH_REG_IDX_CFM         11
#define EZH_REG_IDX_SP          12
#define EZH_REG_IDX_PC          13
#define EZH_REG_IDX_GPI         14
#define EZH_REG_IDX_RA          15
#define EZH_REG_IDX_FLAGS       16

#define EZH_NUM_REGS            17

#endif // LLDB_SYMBOL_EZHREGISTERS_H
