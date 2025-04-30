//===-- Next32RegisterInfo.td - Next32 Register defs -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//  This file describes Next32Silicon virtual register configuration
//===----------------------------------------------------------------------===//

class Reg<bits<16> Enc, string n> : Register<n> {
  let Namespace = "Next32";
  let HWEncoding = Enc;
}

// General purpose registers
foreach I = 1 - 2000 in def R #I : Reg<I, "r" #I>,
        DwarfRegNum<[I]>;

def TID : Reg<0xC000, "tid">, DwarfRegNum<[0xC000]>;
def STACK_SIZE : Reg<0xC001, "stack_size">, DwarfRegNum<[0xC001]>;
def MBB_ADDR : Reg<0xC002, "mbb_addr">, DwarfRegNum<[0xC002]>;
def MBB_ADDR_1 : Reg<0xC003, "mbb_addr_1">, DwarfRegNum<[0xC003]>;
def MBB_ADDR_2 : Reg<0xC004, "mbb_addr_2">, DwarfRegNum<[0xC004]>;
def CALL_ADDR : Reg<0xC005, "call_addr">, DwarfRegNum<[0xC005]>;
def CALL_RET_BB : Reg<0xC006, "call_ret_bb">, DwarfRegNum<[0xC006]>;
def CALL_RET_FID : Reg<0xC007, "call_ret_fid">, DwarfRegNum<[0xC007]>;
def RET_FID : Reg<0xC008, "ret_fid">, DwarfRegNum<[0xC008]>;
def SP_HIGH : Reg<0xC009, "sp_high">, DwarfRegNum<[0xC009]>;
def SP_LOW : Reg<0xC00A, "sp_low">, DwarfRegNum<[0xC00A]>;
def VA_HIGH : Reg<0xC00B, "va_high">, DwarfRegNum<[0xC00B]>;
def VA_LOW : Reg<0xC00C, "va_low">, DwarfRegNum<[0xC00C]>;
def SPREL_HIGH : Reg<0xC00E, "sprel_high">, DwarfRegNum<[0xC00E]>;
def SPREL_LOW : Reg<0xC00F, "sprel_low">, DwarfRegNum<[0xC00F]>;
def STACK_BIAS : Reg<0xC010, "stack_bias">, DwarfRegNum<[0xC010]>;
def SCRATCH1 : Reg<0xC011, "scratch1">, DwarfRegNum<[0xC011]>;
def SCRATCH2 : Reg<0xC012, "scratch2">, DwarfRegNum<[0xC012]>;

def GPR32 : RegisterClass<"Next32", [i32], 32, (sequence "R%u", 1, 2000)>;
def RESERVED
    : RegisterClass<"Next32", [i32], 32,
                    (add TID, STACK_SIZE, MBB_ADDR, MBB_ADDR_1, MBB_ADDR_2,
                     CALL_ADDR, CALL_RET_BB, CALL_RET_FID, RET_FID, SP_HIGH,
                     SP_LOW, VA_HIGH, VA_LOW, SPREL_HIGH, SPREL_LOW, STACK_BIAS,
                     SCRATCH1, SCRATCH2)>;
def FADDR : RegisterClass<"Next32", [i32], 32,
                          (add MBB_ADDR, CALL_ADDR, CALL_RET_BB, CALL_RET_FID)>;
