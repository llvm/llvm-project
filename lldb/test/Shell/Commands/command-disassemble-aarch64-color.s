# UNSUPPORTED: system-windows
# REQUIRES: aarch64

# This checks that lldb's disassembler colors AArch64 disassembly.

# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnueabihf %s -o %t --mattr=+all
# RUN: %lldb %t -o "settings set use-color true" -o "disassemble -n fn" -o exit 2>&1 | FileCheck %s

.globl  fn
.type   fn, @function
fn:
  // These are in alphabetical order by extension name
  aesd v0.16b, v0.16b                   // AEK_AES
  bfadd z23.h, p3/m, z23.h, z13.h       // AEK_B16B16
  bfdot v2.2s, v3.4h, v4.4h             // AEK_BF16
  brb iall                              // AEK_BRBE
  crc32b w0, w0, w0                     // AEK_CRC
  // AEK_CRYPTO enables a combination of other features
  smin x0, x0, #0                       // AEK_CSSC
  sysp	#0, c2, c0, #0, x0, x1          // AEK_D128
  sdot v0.2s, v1.8b, v2.8b              // AEK_DOTPROD
  fmmla z0.s, z1.s, z2.s                // AEK_F32MM

# CHECK: `fn:
# CHECK-NEXT: [0x0] <+0>:    aesd   [0;36mv0[0m.16b, [0;36mv0[0m.16b
# CHECK-NEXT: [0x4] <+4>:    bfadd  [0;36mz23[0m.h, [0;36mp3[0m/m, [0;36mz23[0m.h, [0;36mz13[0m.h
# CHECK-NEXT: [0x8] <+8>:    bfdot  [0;36mv2[0m.2s, [0;36mv3[0m.4h, [0;36mv4[0m.4h
# CHECK-NEXT: [0xc] <+12>:   brb    iall
# CHECK-NEXT: [0x10] <+16>:  crc32b [0;36mw0[0m, [0;36mw0[0m, [0;36mw0[0m
# CHECK-NEXT: [0x14] <+20>:  smin   [0;36mx0[0m, [0;36mx0[0m, [0;31m#0x0[0m
# CHECK-NEXT: [0x18] <+24>:  sysp   [0;31m#0x0[0m, c2, c0, [0;31m#0x0[0m, [0;36mx0[0m, [0;36mx1[0m
# CHECK-NEXT: [0x1c] <+28>:  sdot   [0;36mv0[0m.2s, [0;36mv1[0m.8b, [0;36mv2[0m.8b
# CHECK-NEXT: [0x20] <+32>:  fmmla  [0;36mz0[0m.s, [0;36mz1[0m.s, [0;36mz2[0m.s
