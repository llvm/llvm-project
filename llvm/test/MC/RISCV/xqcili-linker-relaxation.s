# RUN: llvm-mc --triple=riscv32 -mattr=+relax,+experimental-xqcili \
# RUN:    %s -filetype=obj -o - -riscv-add-build-attributes \
# RUN:    | llvm-objdump -dr -M no-aliases - \
# RUN:    | FileCheck %s

## This tests that we correctly emit relocations for linker relaxation when
## emitting `QC.E.LI` and `QC.LI`.

  .section .text.ex1, "ax", @progbits
# CHECK-LABEL: <.text.ex1>:
  blez    a1, .L1
# CHECK-NEXT: bge zero, a1, 0x0 <.text.ex1>
# CHECK-NEXT: R_RISCV_BRANCH .L1{{$}}
  qc.e.li a0, sym
# CHECK-NEXT: qc.e.li a0, 0x0
# CHECK-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# CHECK-NEXT: R_RISCV_CUSTOM194 sym{{$}}
# CHECK-NEXT: R_RISCV_RELAX *ABS*{{$}}
.L1:
# CHECK: <.L1>:
  ret
# CHECK-NEXT: c.jr ra

  .section .text.ex2, "ax", @progbits
# CHECK-LABEL: <.text.ex2>:
  blez    a1, .L2
# CHECK-NEXT: bge zero, a1, 0x0 <.text.ex2>
# CHECK-NEXT: R_RISCV_BRANCH .L2{{$}}
  qc.li a0,  %qc.abs20(sym)
# CHECK-NEXT: qc.li a0, 0x0
# CHECK-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# CHECK-NEXT: R_RISCV_CUSTOM192 sym{{$}}
# CHECK-NEXT: R_RISCV_RELAX *ABS*{{$}}
.L2:
# CHECK: <.L2>:
  ret
# CHECK-NEXT: c.jr ra
