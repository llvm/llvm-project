# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump -d -print-imm-hex %t | tr '\t' ' ' | FileCheck -strict-whitespace %s

# RUN: llvm-objdump -d -print-imm-hex -no-show-raw-insn %t | tr '\t' ' ' | \
# RUN:   FileCheck -check-prefix=NORAW -strict-whitespace %s

# Instructions are expected to be aligned if the instruction in hex is not too long.

# CHECK:       0: c3                            retq
# CHECK-NEXT:  1: 48 8b 05 56 34 12 00          movq 0x123456(%rip), %rax
# CHECK-NEXT:  8: 48 b8 54 55 55 55 55 55 55 55 movabsq $0x5555555555555554, %rax
# CHECK-NEXT: 12: 8f ea 00 12 4c 02 40 00 00 00 00      lwpval $0x0, 0x40(%rdx,%rax), %r15d
# CHECK-NEXT: 1d: 8f ea 00 12 04 25 f0 1c f0 1c 00 00 00 00     lwpins $0x0, 0x1cf01cf0, %r15d
# CHECK-NEXT: 2b: ff ff                         <unknown>

# NORAW:       0:       retq
# NORAW-NEXT:  1:       movq 0x123456(%rip), %rax
# NORAW-NEXT:  8:       movabsq $0x5555555555555554, %rax
# NORAW-NEXT: 12:       lwpval $0x0, 0x40(%rdx,%rax), %r15d
# NORAW-NEXT: 1d:       lwpins $0x0, 0x1cf01cf0, %r15d
# NORAW-NEXT: 2b:       <unknown>

.text
  retq
  movq 0x123456(%rip),%rax
  movabs $0x5555555555555554,%rax
  lwpval $0x0, 0x40(%rdx,%rax), %r15d
  lwpins $0x0, 0x1cf01cf0, %r15d
  .word 0xffff
