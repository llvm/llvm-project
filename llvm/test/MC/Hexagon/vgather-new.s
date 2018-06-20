// RUN: llvm-mc -arch=hexagon -mcpu=hexagonv65 -mhvx -show-encoding %s -o - | FileCheck %s

// TypeCVI_FIRST was set incorrectly, causing vgather not to be considered
// a vector instruction. This resulted in an incorrect encoding of the vtmp.new
// operand in the store.
// CHECK: encoding: [0x1f,0x45,0x05,0x2f,0x22,0xc0,0x21,0x28]

{
  if (q0) vtmp.h = vgather(r5,m0,v31.h).h
  vmem(r1+#0) = vtmp.new
}

