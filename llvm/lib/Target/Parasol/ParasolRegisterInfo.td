//===-- ParasolRegisterInfo.td - Parasol Register defs ---------*- tablegen -*-===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the Parasol register file
//===----------------------------------------------------------------------===//

let Namespace = "Parasol" in {
class ParasolReg<bits<6> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{5-0} = Enc;
  let AltNames = alt;
}

// TODO: Add some registers for the constants zero and one?

// let FallbackRegAltNameIndex = NoRegAltName in
// def ABIRegAltName : RegAltNameIndex;

} // end Namespace

//===----------------------------------------------------------------------===//
//  Integer registers
//===----------------------------------------------------------------------===//
/*
let RegAltNameIndices = [ABIRegAltName] in {
  let isConstant = true in
  def X0  : ParasolReg<0, "x0", ["zero"]>, DwarfRegNum<[0]>;
  let CostPerUse = [0, 1] in {
  def X1  : ParasolReg<1, "x1", ["ra"]>, DwarfRegNum<[1]>;
  def X2  : ParasolReg<2, "x2", ["sp"]>, DwarfRegNum<[2]>;
  def X3  : ParasolReg<3, "x3", ["gp"]>, DwarfRegNum<[3]>;
  def X4  : ParasolReg<4, "x4", ["tp"]>, DwarfRegNum<[4]>;
  def X5  : ParasolReg<5, "x5", ["t0"]>, DwarfRegNum<[5]>;
  def X6  : ParasolReg<6, "x6", ["t1"]>, DwarfRegNum<[6]>;
  def X7  : ParasolReg<7, "x7", ["t2"]>, DwarfRegNum<[7]>;
  }
  def X8  : ParasolReg<8, "x8", ["s0", "fp"]>, DwarfRegNum<[8]>;
  def X9  : ParasolReg<9, "x9", ["s1"]>, DwarfRegNum<[9]>;
  def X10 : ParasolReg<10,"x10", ["a0"]>, DwarfRegNum<[10]>;
  def X11 : ParasolReg<11,"x11", ["a1"]>, DwarfRegNum<[11]>;
  def X12 : ParasolReg<12,"x12", ["a2"]>, DwarfRegNum<[12]>;
  def X13 : ParasolReg<13,"x13", ["a3"]>, DwarfRegNum<[13]>;
  def X14 : ParasolReg<14,"x14", ["a4"]>, DwarfRegNum<[14]>;
  def X15 : ParasolReg<15,"x15", ["a5"]>, DwarfRegNum<[15]>;
  let CostPerUse = [0, 1] in {
  def X16 : ParasolReg<16,"x16", ["a6"]>, DwarfRegNum<[16]>;
  def X17 : ParasolReg<17,"x17", ["a7"]>, DwarfRegNum<[17]>;
  def X18 : ParasolReg<18,"x18", ["s2"]>, DwarfRegNum<[18]>;
  def X19 : ParasolReg<19,"x19", ["s3"]>, DwarfRegNum<[19]>;
  def X20 : ParasolReg<20,"x20", ["s4"]>, DwarfRegNum<[20]>;
  def X21 : ParasolReg<21,"x21", ["s5"]>, DwarfRegNum<[21]>;
  def X22 : ParasolReg<22,"x22", ["s6"]>, DwarfRegNum<[22]>;
  def X23 : ParasolReg<23,"x23", ["s7"]>, DwarfRegNum<[23]>;
  def X24 : ParasolReg<24,"x24", ["s8"]>, DwarfRegNum<[24]>;
  def X25 : ParasolReg<25,"x25", ["s9"]>, DwarfRegNum<[25]>;
  def X26 : ParasolReg<26,"x26", ["s10"]>, DwarfRegNum<[26]>;
  def X27 : ParasolReg<27,"x27", ["s11"]>, DwarfRegNum<[27]>;
  def X28 : ParasolReg<28,"x28", ["t3"]>, DwarfRegNum<[28]>;
  def X29 : ParasolReg<29,"x29", ["t4"]>, DwarfRegNum<[29]>;
  def X30 : ParasolReg<30,"x30", ["t5"]>, DwarfRegNum<[30]>;
  def X31 : ParasolReg<31,"x31", ["t6"]>, DwarfRegNum<[31]>;
  }
}
*/

// Define 64 integer registers using a loop
foreach i = 0-63 in {
  def X#i : ParasolReg<i, "x"#i>;
}

// Add to the integer register class
// NOTE: Due to a bug in LLVM this ValueType list needs to be specified in
// largest to smallest order. Otherwise when put into a register bank during
// globalisel the size of the register bank will be too small (it seems to
// choose the first element as the size of the class).
def IR : RegisterClass<"Parasol", [i32, i16, i8, i1], 32, (sequence "X%u", 0, 63)>;
