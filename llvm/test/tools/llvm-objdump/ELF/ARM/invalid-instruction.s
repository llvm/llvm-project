@RUN: llvm-mc -triple arm-unknown-linux -filetype=obj %s | llvm-objdump -d - | FileCheck %s

.text
  b l0
  .inst 0xffffffff
l0:

@CHECK:            0: ea000000      b 0x8 <l0> @ imm = #0
@CHECK-NEXT:       4: ffffffff     <unknown>
