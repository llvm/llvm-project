//== M68kRegisterInfo.td - M68k register definitions ----*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the M68k Register file, defining the registers
/// aliases between the registers, and the register classes built out of the
/// registers.
///
//===----------------------------------------------------------------------===//

class MxReg<string N, bits<16> ENC,
            list<Register> SUBREGS = [], list<SubRegIndex> SUBIDX,
            list<int> DWREGS = []>
    : Register<N>, DwarfRegNum<DWREGS> {
  let Namespace     = "M68k";
  let HWEncoding    = ENC;
  let SubRegs       = SUBREGS;
  let SubRegIndices = SUBIDX;
}

// Subregister indices.
let Namespace = "M68k" in {
  def MxSubRegIndex8Lo  : SubRegIndex<8, 0>;
  def MxSubRegIndex16Lo : SubRegIndex<16, 0>;
}

// Generate Data registers and theirs smaller variants
foreach Index = 0-7 in {
  def "BD"#Index : MxReg<"d"#Index, Index, [], [], [Index]>;

  def "WD"#Index
    : MxReg<"d"#Index, Index,
            [!cast<Register>("BD"#Index)], [MxSubRegIndex8Lo],
            [Index]>;

  def "D"#Index
    : MxReg<"d"#Index, Index,
            [!cast<Register>("WD"#Index)], [MxSubRegIndex16Lo],
            [Index]>;

} // foreach

// Generate Address registers and theirs smaller variants
foreach Index = 0-7 in {
  def "WA"#Index
     : MxReg<"a"#Index, Index, [], [], [!add(8,Index)]>;

  def "A"#Index
     : MxReg<"a"#Index, Index,
             [!cast<Register>("WA"#Index)], [MxSubRegIndex16Lo],
             [!add(8,Index)]>;
}

// Alias Registers
class MxAliasReg<string N, MxReg REG>
    : MxReg<N, REG.HWEncoding, [], [], REG.DwarfNumbers> {
  let Aliases = [REG];
}

def BP  : MxAliasReg<"bp",  A5>;
def FP  : MxAliasReg<"fp",  A6>;
def SP : MxAliasReg<"sp", A7>;

def USP : MxAliasReg<"usp", A7>;
def SSP : MxAliasReg<"ssp", A7>;
def ISP : MxAliasReg<"isp", A7>;

// Pseudo Registers
class MxPseudoReg<string N, list<Register> SUBREGS = [], list<SubRegIndex> SUBIDX = []>
    : MxReg<N, 0, SUBREGS, SUBIDX>;

def CCR : MxPseudoReg<"ccr">;
def SR  : MxPseudoReg<"sr">;

def PC  : MxPseudoReg<"pc">;

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

class MxRegClass<list<ValueType> regTypes, int alignment, dag regList>
    : RegisterClass<"M68k", regTypes, alignment, regList>;

// Data Registers
def DR8  : MxRegClass<[i8],  16, (sequence "BD%u", 0, 7)>;
def DR16 : MxRegClass<[i16], 16, (sequence "WD%u", 0, 7)>;
def DR32 : MxRegClass<[i32], 32, (sequence "D%u",  0, 7)>;

// Address Registers
def AR16 : MxRegClass<[i16], 16, (sequence "WA%u", 0, 6)>;
def AR32 : MxRegClass<[i32], 32, (add (sequence "A%u", 0, 6), SP)>;

def AR32_NOSP : MxRegClass<[i32], 32, (add (sequence "A%u", 0, 6))>;

// Index Register Classes
// FIXME try alternative ordering like `D0, D1, A0, A1, ...`
def XR16 : MxRegClass<[i16], 16, (add DR16, AR16)>;
def XR32 : MxRegClass<[i32], 32, (add DR32, AR32)>;

def SPC  : MxRegClass<[i32], 32, (add SP)>;

let CopyCost = -1 in {
  def CCRC : MxRegClass<[i8],  16, (add CCR)>;
  def SRC  : MxRegClass<[i16], 16, (add SR)>;
}

let isAllocatable = 0 in {
  def PCC  : MxRegClass<[i32], 32, (add PC)>;
}

// Register used with tail call
def DR16_TC : MxRegClass<[i16], 16, (add D0, D1)>;
def DR32_TC : MxRegClass<[i32], 32, (add D0, D1)>;

def AR16_TC : MxRegClass<[i16], 16, (add A0, A1)>;
def AR32_TC : MxRegClass<[i32], 32, (add A0, A1)>;

def XR16_TC : MxRegClass<[i16], 16, (add DR16_TC, AR16_TC)>;
def XR32_TC : MxRegClass<[i32], 32, (add DR32_TC, AR32_TC)>;

// These classes provide spill/restore order if used with MOVEM instruction
def SPILL   : MxRegClass<[i32], 32, (add (add (sequence "D%u", 0, 7),
                                              (sequence "A%u", 0, 6)), SP)>;
def SPILL_R : MxRegClass<[i32], 32, (add SP, (add (sequence "A%u", 6, 0),
                                                  (sequence "D%u", 7, 0)))>;
