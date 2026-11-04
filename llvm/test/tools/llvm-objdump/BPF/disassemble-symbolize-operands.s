# REQUIRES: bpf-registered-target

## --symbolize-operands is the default for BPF. Verify that it works
## both explicitly and by default, and that --no-symbolize-operands
## disables it.

# RUN: llvm-mc -triple=bpfel %s -filetype=obj -o %t

## Default (implicit --symbolize-operands for BPF).
# RUN: llvm-objdump -d --no-show-raw-insn --no-leading-addr %t | \
# RUN:   FileCheck %s

## Explicit --symbolize-operands.
# RUN: llvm-objdump -d --symbolize-operands --no-show-raw-insn --no-leading-addr %t | \
# RUN:   FileCheck %s

## --no-symbolize-operands disables the default.
# RUN: llvm-objdump -d --no-symbolize-operands --no-show-raw-insn --no-leading-addr %t | \
# RUN:   FileCheck %s --check-prefix=NOSYM

        .text
main:
        if r1 > 42 goto +2
        r1 -= 10
        goto -3
        r0 = 0
        exit

# CHECK:      <main>:
# CHECK-NEXT: <L0>:
# CHECK-NEXT: 	if r1 > 0x2a goto +0x2 <L1>
# CHECK-NEXT: 	r1 -= 0xa
# CHECK-NEXT: 	goto -0x3 <main>
# CHECK-NEXT: <L1>:
# CHECK-NEXT: 	r0 = 0x0
# CHECK-NEXT: 	exit

# NOSYM:      <main>:
# NOSYM-NEXT: 	if r1 > 0x2a goto +0x2
# NOSYM-NEXT: 	r1 -= 0xa
# NOSYM-NEXT: 	goto -0x3
# NOSYM-NEXT: 	r0 = 0x0
# NOSYM-NEXT: 	exit
