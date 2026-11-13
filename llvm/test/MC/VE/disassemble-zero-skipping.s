# RUN: llvm-mc -triple=ve -filetype=obj %s -o %t
# RUN: llvm-objdump --no-print-imm-hex -d %t | FileCheck %s --check-prefixes=CHECK,SKIP --implicit-check-not='<unknown>'
# RUN: llvm-objdump --no-print-imm-hex -d --disassemble-zeroes %t | FileCheck %s --check-prefixes=CHECK,ZERO

# Eight padding zeroes plus NOP's seven leading zeroes must not cause a
# four-byte skip into NOP. Keep NOP unlabeled so symbols do not bound the scan.
# The function branches over the padding and returns 42.

# CHECK-LABEL: <foo>:
# CHECK-NEXT:  0: 10 00 00 00 00 00 3f 18 br.l.t 16
# SKIP-NEXT:   ...
# ZERO-NEXT:   8: 00 00 00 00 00 00 00 00 <unknown>
# CHECK-NEXT: 10: 00 00 00 00 00 00 00 79 nop
# CHECK-NEXT: 18: 2a 00 00 00 00 00 00 06 lea %s0, 42
# CHECK-NEXT: 20: 00 00 00 00 8a 00 3f 19 b.l.t (, %s10)

.text
.p2align 3
.globl foo
.type foo, @function
foo:
  br.l.t 16
  .p2align 4, 0
  nop
  lea %s0, 42
  b.l.t (, %s10)
.size foo, .-foo
