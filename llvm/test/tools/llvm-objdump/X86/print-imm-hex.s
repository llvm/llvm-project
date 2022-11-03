# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t

# RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=PRINT
# RUN: llvm-objdump -d --print-imm-hex --no-print-imm-hex %t | FileCheck %s --check-prefix=NOPRINT
# RUN: llvm-objdump -d --no-print-imm-hex --print-imm-hex %t | FileCheck %s --check-prefix=PRINT

.text
  retq
  movq 0x123456(%rip),%rax
  movabs $0x5555555555555554,%rax
  lwpval $0x0, 0x40(%rdx,%rax), %r15d
  lwpins $0x0, 0x1cf01cf0, %r15d
  .word 0xffff

# NOPRINT:      0000000000000000 <.text>:
# NOPRINT-NEXT:  0: c3                            retq
# NOPRINT-NEXT:  1: 48 8b 05 56 34 12 00          movq    1193046(%rip), %rax  # 0x12345e <.text+0x12345e>
# NOPRINT-NEXT:  8: 48 b8 54 55 55 55 55 55 55 55 movabsq $6148914691236517204, %rax # imm = 0x5555555555555554
# NOPRINT-NEXT: 12: 8f ea 00 12 4c 02 40 00 00 00 00      lwpval  $0, 64(%rdx,%rax), %r15d
# NOPRINT-NEXT: 1d: 8f ea 00 12 04 25 f0 1c f0 1c 00 00 00 00     lwpins  $0, 485498096, %r15d
# NOPRINT-NEXT: 2b: ff ff                         <unknown>

# PRINT:       0000000000000000 <.text>:
# PRINT-NEXT:  0: c3                            retq
# PRINT-NEXT:  1: 48 8b 05 56 34 12 00          movq    0x123456(%rip), %rax  # 0x12345e <.text+0x12345e>
# PRINT-NEXT:  8: 48 b8 54 55 55 55 55 55 55 55 movabsq $0x5555555555555554, %rax # imm = 0x5555555555555554
# PRINT-NEXT: 12: 8f ea 00 12 4c 02 40 00 00 00 00      lwpval  $0x0, 0x40(%rdx,%rax), %r15d
# PRINT-NEXT: 1d: 8f ea 00 12 04 25 f0 1c f0 1c 00 00 00 00     lwpins  $0x0, 0x1cf01cf0, %r15d
# PRINT-NEXT: 2b: ff ff                         <unknown>
