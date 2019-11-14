# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 | FileCheck %s

{ r0=sub(#1,r0)
  r1=sub(#1, r0)
  memw(r0)=r0
  if (p3) dealloc_return }


# CHECK: note: Instruction can utilize slots: 0, 1, 2, 3
# CHECK: note: Instruction can utilize slots: 0, 1, 2, 3
# CHECK: note: Instruction can utilize slots: <None>
# CHECK: note: Instruction can utilize slots: 0
# CHECK: note: Instruction was restricted from being in slot 1
# CHECK: note: Instruction does not allow a store in slot 1
# CHECK: error: invalid instruction packet: slot error
