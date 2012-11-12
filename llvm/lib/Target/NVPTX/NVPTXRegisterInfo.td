//===-- NVPTXRegisterInfo.td - NVPTX Register defs ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the PTX register file
//===----------------------------------------------------------------------===//

class NVPTXReg<string n> : Register<n> {
  let Namespace = "NVPTX";
}

class NVPTXRegClass<list<ValueType> regTypes, int alignment, dag regList>
     : RegisterClass <"NVPTX", regTypes, alignment, regList>;

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//

// Special Registers used as stack pointer
def VRFrame         : NVPTXReg<"%SP">;
def VRFrameLocal    : NVPTXReg<"%SPL">;

// Special Registers used as the stack
def VRDepot  : NVPTXReg<"%Depot">;

foreach i = 0-395 in {
  def P#i  : NVPTXReg<"%p"#i>;  // Predicate
  def RC#i : NVPTXReg<"%rc"#i>; // 8-bit
  def RS#i : NVPTXReg<"%rs"#i>; // 16-bit
  def R#i  : NVPTXReg<"%r"#i>;  // 32-bit
  def RL#i : NVPTXReg<"%rl"#i>; // 64-bit
  def F#i  : NVPTXReg<"%f"#i>;  // 32-bit float
  def FL#i : NVPTXReg<"%fl"#i>; // 64-bit float
  // Vectors
  foreach s = [ "2b8", "2b16", "2b32", "2b64", "4b8", "4b16", "4b32" ] in
    def v#s#_#i : NVPTXReg<"%v"#s#"_"#i>;

  // Arguments
  def ia#i : NVPTXReg<"%ia"#i>;
  def la#i : NVPTXReg<"%la"#i>;
  def fa#i : NVPTXReg<"%fa"#i>;
  def da#i : NVPTXReg<"%da"#i>;
}

//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//
def Int1Regs : NVPTXRegClass<[i1], 8, (add (sequence "P%u", 0, 395))>;
def Int8Regs : NVPTXRegClass<[i8], 8, (add (sequence "RC%u", 0, 395))>;
def Int16Regs : NVPTXRegClass<[i16], 16, (add (sequence "RS%u", 0, 395))>;
def Int32Regs : NVPTXRegClass<[i32], 32, (add (sequence "R%u", 0, 395))>;
def Int64Regs : NVPTXRegClass<[i64], 64, (add (sequence "RL%u", 0, 395))>;
def Float32Regs : NVPTXRegClass<[f32], 32, (add (sequence "F%u", 0, 395))>;
def Float64Regs : NVPTXRegClass<[f64], 64, (add (sequence "FL%u", 0, 395))>;
def Int32ArgRegs : NVPTXRegClass<[i32], 32, (add (sequence "ia%u", 0, 395))>;
def Int64ArgRegs : NVPTXRegClass<[i64], 64, (add (sequence "la%u", 0, 395))>;
def Float32ArgRegs : NVPTXRegClass<[f32], 32, (add (sequence "fa%u", 0, 395))>;
def Float64ArgRegs : NVPTXRegClass<[f64], 64, (add (sequence "da%u", 0, 395))>;

// Read NVPTXRegisterInfo.cpp to see how VRFrame and VRDepot are used.
def SpecialRegs : NVPTXRegClass<[i32], 32, (add VRFrame, VRDepot)>;

class NVPTXVecRegClass<list<ValueType> regTypes, int alignment, dag regList,
                       NVPTXRegClass sClass,
                       int e,
                       string n>
  : NVPTXRegClass<regTypes, alignment, regList>
{
  NVPTXRegClass scalarClass=sClass;
  int elems=e;
  string name=n;
}
def V2F32Regs
  : NVPTXVecRegClass<[v2f32], 64, (add (sequence "v2b32_%u", 0, 395)),
    Float32Regs, 2, ".v2.f32">;
def V4F32Regs
  : NVPTXVecRegClass<[v4f32], 128, (add (sequence "v4b32_%u", 0, 395)),
    Float32Regs, 4, ".v4.f32">;
def V2I32Regs
  : NVPTXVecRegClass<[v2i32], 64, (add (sequence "v2b32_%u", 0, 395)),
    Int32Regs, 2, ".v2.u32">;
def V4I32Regs
  : NVPTXVecRegClass<[v4i32], 128, (add (sequence "v4b32_%u", 0, 395)),
    Int32Regs, 4, ".v4.u32">;
def V2F64Regs
  : NVPTXVecRegClass<[v2f64], 128, (add (sequence "v2b64_%u", 0, 395)),
    Float64Regs, 2, ".v2.f64">;
def V2I64Regs
  : NVPTXVecRegClass<[v2i64], 128, (add (sequence "v2b64_%u", 0, 395)),
    Int64Regs, 2, ".v2.u64">;
def V2I16Regs
  : NVPTXVecRegClass<[v2i16], 32, (add (sequence "v2b16_%u", 0, 395)),
    Int16Regs, 2, ".v2.u16">;
def V4I16Regs
  : NVPTXVecRegClass<[v4i16], 64, (add (sequence "v4b16_%u", 0, 395)),
    Int16Regs, 4, ".v4.u16">;
def V2I8Regs
  : NVPTXVecRegClass<[v2i8], 16, (add (sequence "v2b8_%u", 0, 395)),
    Int8Regs, 2, ".v2.u8">;
def V4I8Regs
  : NVPTXVecRegClass<[v4i8], 32, (add (sequence "v4b8_%u", 0, 395)),
    Int8Regs, 4, ".v4.u8">;
