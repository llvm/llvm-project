# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s

# Verify that the .align directive emits the proper insn packets.

{ r1 = sub(#1, r1) }
# CHECK: 76414021 { r1 = sub(#1, r1)
# CHECK-NEXT: 7f004000   nop
# CHECK-NEXT: 7f004000   nop
# CHECK-NEXT: 7f00c000   nop }

.align 16
{ r1 = sub(#1, r1)
  r2 = sub(#1, r2) }
# CHECK: 76414021 { r1 = sub(#1, r1)
# CHECK-NEXT: 76424022   r2 = sub(#1, r2)
# CHECK-NEXT: 7f004000  nop
# CHECK-NEXT: 7f00c000   nop }

.p2align 5
{ r1 = sub(#1, r1)
  r2 = sub(#1, r2)
  r3 = sub(#1, r3) }
# CHECK: 76434023   r3 = sub(#1, r3)
# CHECK-NEXT: 7f00c000 nop }

.align 16
{ r1 = sub(#1, r1)
  r2 = sub(#1, r2)
  r3 = sub(#1, r3)
  r4 = sub(#1, r4) }

# Don't pad packets that can't be padded e.g. solo insts
# CHECK: 9200c020 {  r0 = vextract(v0,r0) }
r0 = vextract(v0, r0)
.align 128
# CHECK: 76414021 { r1 = sub(#1, r1)
# CHECK-NEXT: 7f00c000   nop }
{ r1 = sub(#1, r1) }

#CHECK: { r1 = sub(#1, r1)
#CHECK:   r2 = sub(#1, r2)
#CHECK:   r3 = sub(#1, r3) }
.falign
.align 8
{ r1 = sub(#1, r1)
  r2 = sub(#1, r2)
  r3 = sub(#1, r3)  }

# CHECK: { immext(#0)
# CHECK:   r0 = sub(##1, r0)
# CHECK:   immext(#0)
# CHECK:   r1 = sub(##1, r1) }
# CHECK: { nop
# CHECK:   nop
# CHECK:   nop }
# CHECK: { r0 = sub(#1, r0) }
{ r0 = sub (##1, r0)
  r1 = sub (##1, r1) }
.align 16
{ r0 = sub (#1, r0) }