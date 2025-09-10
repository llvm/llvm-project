/// Checks that various hexagon scenarios are handled correctly:
///  - branch targets
///  - endloops
///  - inline-relocs
///  - multi-insn bundles

{
  r6 = sub(r1, r0)
  r7 = and(r4, #0x0)
  if (p1) jump:t target1
  if (p2) jump:nt target2
}

{
  r8 = r7
  r9 = add(r8, #0)
  r10 = memw(r9)
} :endloop0

{ jump ##sym }

target1:
  nop

target2:
  nop

// RUN: llvm-mc %s --triple=hexagon -filetype=obj | llvm-objdump -d -r - | FileCheck %s

//      CHECK: 00000000 <.text>:
// CHECK-NEXT:        0:       12 51 00 5c     5c005112 {      if (p1) jump:t 0x24 <target1>
// CHECK-NEXT:        4:       14 42 00 5c     5c004214        if (p2) jump:nt 0x28 <target2>
// CHECK-NEXT:        8:       06 41 20 f3     f3204106        r6 = sub(r1,r0)
// CHECK-NEXT:        c:       07 c0 04 76     7604c007        r7 = and(r4,#0x0) }
// CHECK-NEXT:       10:       08 80 67 70     70678008 {      r8 = r7
// CHECK-NEXT:       14:       09 40 08 b0     b0084009        r9 = add(r8,#0x0)
// CHECK-NEXT:       18:       0a c0 89 91     9189c00a        r10 = memw(r9+#0x0) }  :endloop0
// CHECK-NEXT:       1c:       00 40 00 00     00004000 {      immext(#0x0)
// CHECK-NEXT:                         0000001c:  R_HEX_B32_PCREL_X    sym
// CHECK-NEXT:       20:       00 c0 00 58     5800c000        jump 0x1c <.text+0x1c> }
// CHECK-NEXT:                         00000020:  R_HEX_B22_PCREL_X    sym+0x4
// CHECK-EMPTY:
// CHECK-NEXT: 00000024 <target1>:
// CHECK-NEXT:       24:       00 c0 00 7f     7f00c000 {      nop }
// CHECK-EMPTY:
// CHECK-NEXT: 00000028 <target2>:
// CHECK-NEXT:       28:       00 c0 00 7f     7f00c000 {      nop }
