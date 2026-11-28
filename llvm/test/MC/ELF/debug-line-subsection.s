# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-objdump -d --no-show-raw-insn %t.o | FileCheck %s --check-prefix=DIS
# RUN: llvm-dwarfdump --debug-line %t.o | FileCheck %s --check-prefix=LOC

# RUN: llvm-mc -filetype=obj -triple=x86_64 -g --defsym GEN=1 %s -o %t.g.o
# RUN: llvm-dwarfdump --debug-line --debug-info --debug-aranges %t.g.o | \
# RUN:   FileCheck %s --check-prefix=GEN
# RUN: llvm-dwarfdump --verify %t.g.o | FileCheck %s --check-prefix=VERIFY

# VERIFY: No errors.

# DIS:      0: nop
# DIS-NEXT: 1: nop
# DIS-NEXT: 2: ret
# DIS-NEXT: 3: int3
# DIS-NEXT: 4: cld
# DIS-NEXT: 5: cmc
# DIS-NEXT: 6: hlt
# DIS-NEXT: 7: std

# LOC:      Address            Line
# LOC-NEXT: ---
# LOC-NEXT: 0x0000000000000000      1
# LOC-NEXT: 0x0000000000000001      5
# LOC-NEXT: 0x0000000000000002      8
# LOC-NEXT: 0x0000000000000003      3
# LOC-NEXT: 0x0000000000000004      4
# LOC-NEXT: 0x0000000000000005      7
# LOC-NEXT: 0x0000000000000006      2
# LOC-NEXT: 0x0000000000000007      6
# LOC-NEXT: 0x0000000000000008      6 {{.*}} end_sequence
# LOC-EMPTY:

# GEN:      DW_TAG_compile_unit
# GEN:        DW_AT_low_pc (0x0000000000000000)
# GEN-NEXT:   DW_AT_high_pc (0x0000000000000008)

# GEN:      Address Range Header:
# GEN-NEXT: [0x0000000000000000, 0x0000000000000008)

# GEN:      Address            Line
# GEN-NEXT: ---
# GEN-NEXT: 0x0000000000000000
# GEN-NEXT: 0x0000000000000001
# GEN-NEXT: 0x0000000000000002
# GEN-NEXT: 0x0000000000000003
# GEN-NEXT: 0x0000000000000004
# GEN-NEXT: 0x0000000000000005
# GEN-NEXT: 0x0000000000000006
# GEN-NEXT: 0x0000000000000007
# GEN-NEXT: 0x0000000000000008 {{.*}} end_sequence
# GEN-EMPTY:

.macro loc line
.ifndef GEN
  .loc 1 \line
.endif
.endm

.ifndef GEN
  .file 1 "a.c"
.endif

  .text
  .globl f
f:
  loc 1
  nop                 # 0x0
  .subsection 2
  loc 2
  hlt                 # 0x6
  .subsection 1
  loc 3
  int3                # 0x3
  loc 4
  cld                 # 0x4
  .subsection 0
  loc 5
  nop                 # 0x1
  .text 2
  loc 6
  std                 # 0x7
  .subsection 1
  loc 7
  cmc                 # 0x5
  .text
  loc 8
  ret                 # 0x2
