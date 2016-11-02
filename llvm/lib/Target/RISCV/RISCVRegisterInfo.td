//===-- RISCVRegisterInfo.td - RISC-V Register defs --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the RISC-V register file
//===----------------------------------------------------------------------===//

let Namespace = "RISCV" in {
  def sub_32 : SubRegIndex<32>;

  class RISCVReg32<bits<5> Enc, string n, list<string> alt = []> : Register<n> {
    let HWEncoding{4-0} = Enc;
    let AltNames = alt;
  }

  // RISCV64 registers don't define an AsmName or AltName. If they specified
  // names aliasing the RISCVReg32 registers, the generation of the default
  // MatchRegisterName/MatchRegisterAltName would fail. When necessary,
  // RISCVAsmParser will need to convert a register number from a RISCVReg32
  // to the equivalent RISCVReg64.
  class RISCVReg64<RISCVReg32 subreg> : Register<""> {
    let HWEncoding{4-0} = subreg.HWEncoding{4-0};
    let SubRegs = [subreg];
    let SubRegIndices = [sub_32];
  }

  def ABIRegAltName : RegAltNameIndex;
}

// Integer registers
let RegAltNameIndices = [ABIRegAltName] in {
  def X0_32    : RISCVReg32<0, "x0", ["zero"]>, DwarfRegNum<[0]>;
  def X1_32    : RISCVReg32<1, "x1", ["ra"]>, DwarfRegNum<[1]>;
  def X2_32    : RISCVReg32<2, "x2", ["sp"]>, DwarfRegNum<[2]>;
  def X3_32    : RISCVReg32<3, "x3", ["gp"]>, DwarfRegNum<[3]>;
  def X4_32    : RISCVReg32<4, "x4", ["tp"]>, DwarfRegNum<[4]>;
  def X5_32    : RISCVReg32<5, "x5", ["t0"]>, DwarfRegNum<[5]>;
  def X6_32    : RISCVReg32<6, "x6", ["t1"]>, DwarfRegNum<[6]>;
  def X7_32    : RISCVReg32<7, "x7", ["t2"]>, DwarfRegNum<[7]>;
  def X8_32    : RISCVReg32<8, "x8", ["s0"]>, DwarfRegNum<[8]>;
  def X9_32    : RISCVReg32<9, "x9", ["s1"]>, DwarfRegNum<[9]>;
  def X10_32   : RISCVReg32<10,"x10", ["a0"]>, DwarfRegNum<[10]>;
  def X11_32   : RISCVReg32<11,"x11", ["a1"]>, DwarfRegNum<[11]>;
  def X12_32   : RISCVReg32<12,"x12", ["a2"]>, DwarfRegNum<[12]>;
  def X13_32   : RISCVReg32<13,"x13", ["a3"]>, DwarfRegNum<[13]>;
  def X14_32   : RISCVReg32<14,"x14", ["a4"]>, DwarfRegNum<[14]>;
  def X15_32   : RISCVReg32<15,"x15", ["a5"]>, DwarfRegNum<[15]>;
  def X16_32   : RISCVReg32<16,"x16", ["a6"]>, DwarfRegNum<[16]>;
  def X17_32   : RISCVReg32<17,"x17", ["a7"]>, DwarfRegNum<[17]>;
  def X18_32   : RISCVReg32<18,"x18", ["s2"]>, DwarfRegNum<[18]>;
  def X19_32   : RISCVReg32<19,"x19", ["s3"]>, DwarfRegNum<[19]>;
  def X20_32   : RISCVReg32<20,"x20", ["s4"]>, DwarfRegNum<[20]>;
  def X21_32   : RISCVReg32<21,"x21", ["s5"]>, DwarfRegNum<[21]>;
  def X22_32   : RISCVReg32<22,"x22", ["s6"]>, DwarfRegNum<[22]>;
  def X23_32   : RISCVReg32<23,"x23", ["s7"]>, DwarfRegNum<[23]>;
  def X24_32   : RISCVReg32<24,"x24", ["s8"]>, DwarfRegNum<[24]>;
  def X25_32   : RISCVReg32<25,"x25", ["s9"]>, DwarfRegNum<[25]>;
  def X26_32   : RISCVReg32<26,"x26", ["s10"]>, DwarfRegNum<[26]>;
  def X27_32   : RISCVReg32<27,"x27", ["s11"]>, DwarfRegNum<[27]>;
  def X28_32   : RISCVReg32<28,"x28", ["t3"]>, DwarfRegNum<[28]>;
  def X29_32   : RISCVReg32<29,"x29", ["t4"]>, DwarfRegNum<[29]>;
  def X30_32   : RISCVReg32<30,"x30", ["t5"]>, DwarfRegNum<[30]>;
  def X31_32   : RISCVReg32<31,"x31", ["t6"]>, DwarfRegNum<[31]>;
}

foreach Index = 0-31 in {
  def X#Index#_64 : RISCVReg64<!cast<RISCVReg32>("X"#Index#"_32")>, DwarfRegNum<[Index]>;
}

// We currently define separate register classes for the 32-bit and 64-bit
// GPRs. Once variable-sized register classes
// <http://lists.llvm.org/pipermail/llvm-dev/2016-September/105027.html> or
// similar are implemented, we can just use one 'GPR' class for most
// instruction definitions.

// TODO: once codegen is implemented, registers should be listed in an order
// reflecting the preferred register allocation sequence.
def GPR : RegisterClass<"RISCV", [i32], 32, (add
  (sequence "X%u_32", 0, 31)
)>;

def GPR64 : RegisterClass<"RISCV", [i64], 64, (add
  (sequence "X%u_64", 0, 31)
)>;
