//===-- HexagonRegisterInfo.td - Hexagon Register defs -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the Hexagon register file.
//===----------------------------------------------------------------------===//

let Namespace = "Hexagon" in {

  class HexagonReg<string n> : Register<n> {
    field bits<5> Num;
  }

  class HexagonDoubleReg<string n, list<Register> subregs> :
        RegisterWithSubRegs<n, subregs> {
    field bits<5> Num;
  }

  // Registers are identified with 5-bit ID numbers.
  // Ri - 32-bit integer registers.
  class Ri<bits<5> num, string n> : HexagonReg<n> {
    let Num = num;
  }

  // Rf - 32-bit floating-point registers.
  class Rf<bits<5> num, string n> : HexagonReg<n> {
    let Num = num;
  }


  // Rd - 64-bit registers.
  class Rd<bits<5> num, string n, list<Register> subregs> :
        HexagonDoubleReg<n, subregs> {
    let Num = num;
    let SubRegs = subregs;
  }

  // Rp - predicate registers
  class Rp<bits<5> num, string n> : HexagonReg<n> {
    let Num = num;
  }

  // Rc - control registers
  class Rc<bits<5> num, string n> : HexagonReg<n> {
    let Num = num;
  }

  // Rj - aliased integer registers
  class Rj<string n, Ri R>: HexagonReg<n> {
    let Num = R.Num;
    let Aliases = [R];
  }

  def subreg_loreg  : SubRegIndex;
  def subreg_hireg  : SubRegIndex;

  // Integer registers.
  def R0 : Ri< 0, "r0">, DwarfRegNum<[0]>;
  def R1 : Ri< 1, "r1">, DwarfRegNum<[1]>;
  def R2 : Ri< 2, "r2">, DwarfRegNum<[2]>;
  def R3 : Ri< 3, "r3">, DwarfRegNum<[3]>;
  def R4 : Ri< 4, "r4">, DwarfRegNum<[4]>;
  def R5 : Ri< 5, "r5">, DwarfRegNum<[5]>;
  def R6 : Ri< 6, "r6">, DwarfRegNum<[6]>;
  def R7 : Ri< 7, "r7">, DwarfRegNum<[7]>;
  def R8 : Ri< 8, "r8">, DwarfRegNum<[8]>;
  def R9 : Ri< 9, "r9">, DwarfRegNum<[9]>;
  def R10 : Ri<10, "r10">, DwarfRegNum<[10]>;
  def R11 : Ri<11, "r11">, DwarfRegNum<[11]>;
  def R12 : Ri<12, "r12">, DwarfRegNum<[12]>;
  def R13 : Ri<13, "r13">, DwarfRegNum<[13]>;
  def R14 : Ri<14, "r14">, DwarfRegNum<[14]>;
  def R15 : Ri<15, "r15">, DwarfRegNum<[15]>;
  def R16 : Ri<16, "r16">, DwarfRegNum<[16]>;
  def R17 : Ri<17, "r17">, DwarfRegNum<[17]>;
  def R18 : Ri<18, "r18">, DwarfRegNum<[18]>;
  def R19 : Ri<19, "r19">, DwarfRegNum<[19]>;
  def R20 : Ri<20, "r20">, DwarfRegNum<[20]>;
  def R21 : Ri<21, "r21">, DwarfRegNum<[21]>;
  def R22 : Ri<22, "r22">, DwarfRegNum<[22]>;
  def R23 : Ri<23, "r23">, DwarfRegNum<[23]>;
  def R24 : Ri<24, "r24">, DwarfRegNum<[24]>;
  def R25 : Ri<25, "r25">, DwarfRegNum<[25]>;
  def R26 : Ri<26, "r26">, DwarfRegNum<[26]>;
  def R27 : Ri<27, "r27">, DwarfRegNum<[27]>;
  def R28 : Ri<28, "r28">, DwarfRegNum<[28]>;
  def R29 : Ri<29, "r29">, DwarfRegNum<[29]>;
  def R30 : Ri<30, "r30">, DwarfRegNum<[30]>;
  def R31 : Ri<31, "r31">, DwarfRegNum<[31]>;

  def SP : Rj<"sp", R29>, DwarfRegNum<[29]>;
  def FP : Rj<"fp", R30>, DwarfRegNum<[30]>;
  def LR : Rj<"lr", R31>, DwarfRegNum<[31]>;

  // Aliases of the R* registers used to hold 64-bit int values (doubles).
  let SubRegIndices = [subreg_loreg, subreg_hireg], CoveredBySubRegs = 1 in {
  def D0  : Rd< 0,  "r1:0",  [R0,  R1]>, DwarfRegNum<[32]>;
  def D1  : Rd< 2,  "r3:2",  [R2,  R3]>, DwarfRegNum<[34]>;
  def D2  : Rd< 4,  "r5:4",  [R4,  R5]>, DwarfRegNum<[36]>;
  def D3  : Rd< 6,  "r7:6",  [R6,  R7]>, DwarfRegNum<[38]>;
  def D4  : Rd< 8,  "r9:8",  [R8,  R9]>, DwarfRegNum<[40]>;
  def D5  : Rd<10, "r11:10", [R10, R11]>, DwarfRegNum<[42]>;
  def D6  : Rd<12, "r13:12", [R12, R13]>, DwarfRegNum<[44]>;
  def D7  : Rd<14, "r15:14", [R14, R15]>, DwarfRegNum<[46]>;
  def D8  : Rd<16, "r17:16", [R16, R17]>, DwarfRegNum<[48]>;
  def D9  : Rd<18, "r19:18", [R18, R19]>, DwarfRegNum<[50]>;
  def D10 : Rd<20, "r21:20", [R20, R21]>, DwarfRegNum<[52]>;
  def D11 : Rd<22, "r23:22", [R22, R23]>, DwarfRegNum<[54]>;
  def D12 : Rd<24, "r25:24", [R24, R25]>, DwarfRegNum<[56]>;
  def D13 : Rd<26, "r27:26", [R26, R27]>, DwarfRegNum<[58]>;
  def D14 : Rd<28, "r29:28", [R28, R29]>, DwarfRegNum<[60]>;
  def D15 : Rd<30, "r31:30", [R30, R31]>, DwarfRegNum<[62]>;
  }

  // Predicate registers.
  def P0 : Rp<0, "p0">, DwarfRegNum<[63]>;
  def P1 : Rp<1, "p1">, DwarfRegNum<[64]>;
  def P2 : Rp<2, "p2">, DwarfRegNum<[65]>;
  def P3 : Rp<3, "p3">, DwarfRegNum<[66]>;

  // Control registers.
  def SA0 : Rc<0, "sa0">, DwarfRegNum<[67]>;
  def LC0 : Rc<1, "lc0">, DwarfRegNum<[68]>;

  def SA1 : Rc<2, "sa1">, DwarfRegNum<[69]>;
  def LC1 : Rc<3, "lc1">, DwarfRegNum<[70]>;

  def M0 : Rc<6, "m0">, DwarfRegNum<[71]>;
  def M1 : Rc<7, "m1">, DwarfRegNum<[72]>;

  def PC : Rc<9,  "pc">, DwarfRegNum<[32]>; // is the Dwarf number correct?
  def GP : Rc<11, "gp">, DwarfRegNum<[33]>; // is the Dwarf number correct?
}

// Register classes.
//
// FIXME: the register order should be defined in terms of the preferred
// allocation order...
//
def IntRegs : RegisterClass<"Hexagon", [i32,f32], 32,
                            (add (sequence "R%u", 0, 9),
                                 (sequence "R%u", 12, 28),
                                 R10, R11, R29, R30, R31)> {
}

def DoubleRegs : RegisterClass<"Hexagon", [i64,f64], 64,
                               (add (sequence "D%u", 0, 4),
                                    (sequence "D%u", 6, 13), D5, D14, D15)>;


def PredRegs : RegisterClass<"Hexagon", [i1], 32, (add (sequence "P%u", 0, 3))>
{
  let Size = 32;
}

def CRRegs : RegisterClass<"Hexagon", [i32], 32,
                           (add (sequence "LC%u", 0, 1),
                                (sequence "SA%u", 0, 1),
                                (sequence "M%u", 0, 1), PC, GP)> {
  let Size = 32;
}
