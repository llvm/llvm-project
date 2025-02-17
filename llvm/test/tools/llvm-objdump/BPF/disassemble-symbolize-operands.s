# REQUIRES: bpf-registered-target

## Verify generation of 'Lxx' labels for local jump targets, when
## --symbolize-operands option is specified.

# RUN: llvm-mc -triple=bpfel %s -filetype=obj -o %t
# RUN: llvm-objdump -d --symbolize-operands --no-show-raw-insn --no-leading-addr %t | \
# RUN:   FileCheck %s
        .text
main:
        if r1 > 42 goto +2
        r1 -= 10
        goto -3
        r0 = 0
        exit

# CHECK:      <main>:
# CHECK-NEXT: <L1>:
# CHECK-NEXT: 	if r1 > 0x2a goto +0x2 <L0>
# CHECK-NEXT: 	r1 -= 0xa
# CHECK-NEXT: 	goto -0x3 <main>
# CHECK-NEXT: <L0>:
# CHECK-NEXT: 	r0 = 0x0
# CHECK-NEXT: 	exit
