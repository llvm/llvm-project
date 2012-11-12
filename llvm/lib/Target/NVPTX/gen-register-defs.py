#!/usr/bin/env python

num_regs = 396

outFile = open('NVPTXRegisterInfo.td', 'w')

outFile.write('''
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
''')

# Predicates
outFile.write('''
//===--- Predicate --------------------------------------------------------===//
''')
for i in range(0, num_regs):
  outFile.write('def P%d : NVPTXReg<"%%p%d">;\n' % (i, i))

# Int8
outFile.write('''
//===--- 8-bit ------------------------------------------------------------===//
''')
for i in range(0, num_regs):
  outFile.write('def RC%d : NVPTXReg<"%%rc%d">;\n' % (i, i))

# Int16
outFile.write('''
//===--- 16-bit -----------------------------------------------------------===//
''')
for i in range(0, num_regs):
  outFile.write('def RS%d : NVPTXReg<"%%rs%d">;\n' % (i, i))

# Int32
outFile.write('''
//===--- 32-bit -----------------------------------------------------------===//
''')
for i in range(0, num_regs):
  outFile.write('def R%d : NVPTXReg<"%%r%d">;\n' % (i, i))

# Int64
outFile.write('''
//===--- 64-bit -----------------------------------------------------------===//
''')
for i in range(0, num_regs):
  outFile.write('def RL%d : NVPTXReg<"%%rl%d">;\n' % (i, i))

# F32
outFile.write('''
//===--- 32-bit float -----------------------------------------------------===//
''')
for i in range(0, num_regs):
  outFile.write('def F%d : NVPTXReg<"%%f%d">;\n' % (i, i))

# F64
outFile.write('''
//===--- 64-bit float -----------------------------------------------------===//
''')
for i in range(0, num_regs):
  outFile.write('def FL%d : NVPTXReg<"%%fl%d">;\n' % (i, i))

# Vector registers
outFile.write('''
//===--- Vector -----------------------------------------------------------===//
''')
for i in range(0, num_regs):
  outFile.write('def v2b8_%d : NVPTXReg<"%%v2b8_%d">;\n' % (i, i))
for i in range(0, num_regs):
  outFile.write('def v2b16_%d : NVPTXReg<"%%v2b16_%d">;\n' % (i, i))
for i in range(0, num_regs):
  outFile.write('def v2b32_%d : NVPTXReg<"%%v2b32_%d">;\n' % (i, i))
for i in range(0, num_regs):
  outFile.write('def v2b64_%d : NVPTXReg<"%%v2b64_%d">;\n' % (i, i))

for i in range(0, num_regs):
  outFile.write('def v4b8_%d : NVPTXReg<"%%v4b8_%d">;\n' % (i, i))
for i in range(0, num_regs):
  outFile.write('def v4b16_%d : NVPTXReg<"%%v4b16_%d">;\n' % (i, i))
for i in range(0, num_regs):
  outFile.write('def v4b32_%d : NVPTXReg<"%%v4b32_%d">;\n' % (i, i))

# Argument registers
outFile.write('''
//===--- Arguments --------------------------------------------------------===//
''')
for i in range(0, num_regs):
  outFile.write('def ia%d : NVPTXReg<"%%ia%d">;\n' % (i, i))
for i in range(0, num_regs):
  outFile.write('def la%d : NVPTXReg<"%%la%d">;\n' % (i, i))
for i in range(0, num_regs):
  outFile.write('def fa%d : NVPTXReg<"%%fa%d">;\n' % (i, i))
for i in range(0, num_regs):
  outFile.write('def da%d : NVPTXReg<"%%da%d">;\n' % (i, i))

outFile.write('''
//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//
''')

outFile.write('def Int1Regs : NVPTXRegClass<[i1], 8, (add (sequence "P%%u", 0, %d))>;\n' % (num_regs-1))
outFile.write('def Int8Regs : NVPTXRegClass<[i8], 8, (add (sequence "RC%%u", 0, %d))>;\n' % (num_regs-1))
outFile.write('def Int16Regs : NVPTXRegClass<[i16], 16, (add (sequence "RS%%u", 0, %d))>;\n' % (num_regs-1))
outFile.write('def Int32Regs : NVPTXRegClass<[i32], 32, (add (sequence "R%%u", 0, %d))>;\n' % (num_regs-1))
outFile.write('def Int64Regs : NVPTXRegClass<[i64], 64, (add (sequence "RL%%u", 0, %d))>;\n' % (num_regs-1))

outFile.write('def Float32Regs : NVPTXRegClass<[f32], 32, (add (sequence "F%%u", 0, %d))>;\n' % (num_regs-1))
outFile.write('def Float64Regs : NVPTXRegClass<[f64], 64, (add (sequence "FL%%u", 0, %d))>;\n' % (num_regs-1))

outFile.write('def Int32ArgRegs : NVPTXRegClass<[i32], 32, (add (sequence "ia%%u", 0, %d))>;\n' % (num_regs-1))
outFile.write('def Int64ArgRegs : NVPTXRegClass<[i64], 64, (add (sequence "la%%u", 0, %d))>;\n' % (num_regs-1))
outFile.write('def Float32ArgRegs : NVPTXRegClass<[f32], 32, (add (sequence "fa%%u", 0, %d))>;\n' % (num_regs-1))
outFile.write('def Float64ArgRegs : NVPTXRegClass<[f64], 64, (add (sequence "da%%u", 0, %d))>;\n' % (num_regs-1))

outFile.write('''
// Read NVPTXRegisterInfo.cpp to see how VRFrame and VRDepot are used.
def SpecialRegs : NVPTXRegClass<[i32], 32, (add VRFrame, VRDepot)>;
''')

outFile.write('''
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
''')


outFile.write('def V2F32Regs\n  : NVPTXVecRegClass<[v2f32], 64, (add (sequence "v2b32_%%u", 0, %d)),\n    Float32Regs, 2, ".v2.f32">;\n' % (num_regs-1))
outFile.write('def V4F32Regs\n  : NVPTXVecRegClass<[v4f32], 128, (add (sequence "v4b32_%%u", 0, %d)),\n    Float32Regs, 4, ".v4.f32">;\n' % (num_regs-1))

outFile.write('def V2I32Regs\n  : NVPTXVecRegClass<[v2i32], 64, (add (sequence "v2b32_%%u", 0, %d)),\n    Int32Regs, 2, ".v2.u32">;\n' % (num_regs-1))
outFile.write('def V4I32Regs\n  : NVPTXVecRegClass<[v4i32], 128, (add (sequence "v4b32_%%u", 0, %d)),\n    Int32Regs, 4, ".v4.u32">;\n' % (num_regs-1))

outFile.write('def V2F64Regs\n  : NVPTXVecRegClass<[v2f64], 128, (add (sequence "v2b64_%%u", 0, %d)),\n    Float64Regs, 2, ".v2.f64">;\n' % (num_regs-1))
outFile.write('def V2I64Regs\n  : NVPTXVecRegClass<[v2i64], 128, (add (sequence "v2b64_%%u", 0, %d)),\n    Int64Regs, 2, ".v2.u64">;\n' % (num_regs-1))

outFile.write('def V2I16Regs\n  : NVPTXVecRegClass<[v2i16], 32, (add (sequence "v2b16_%%u", 0, %d)),\n    Int16Regs, 2, ".v2.u16">;\n' % (num_regs-1))
outFile.write('def V4I16Regs\n  : NVPTXVecRegClass<[v4i16], 64, (add (sequence "v4b16_%%u", 0, %d)),\n    Int16Regs, 4, ".v4.u16">;\n' % (num_regs-1))

outFile.write('def V2I8Regs\n  : NVPTXVecRegClass<[v2i8], 16, (add (sequence "v2b8_%%u", 0, %d)),\n    Int8Regs, 2, ".v2.u8">;\n' % (num_regs-1))
outFile.write('def V4I8Regs\n  : NVPTXVecRegClass<[v4i8], 32, (add (sequence "v4b8_%%u", 0, %d)),\n    Int8Regs, 4, ".v4.u8">;\n' % (num_regs-1))

outFile.close()


outFile = open('NVPTXNumRegisters.h', 'w')
outFile.write('''
//===-- NVPTXNumRegisters.h - PTX Register Info ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_NUM_REGISTERS_H
#define NVPTX_NUM_REGISTERS_H

namespace llvm {

const unsigned NVPTXNumRegisters = %d;

}

#endif
''' % num_regs)

outFile.close()
