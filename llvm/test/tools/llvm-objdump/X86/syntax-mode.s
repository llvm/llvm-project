## Test att and intel syntax modes.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=ATT
# RUN: llvm-objdump -d -M att %t | FileCheck %s --check-prefix=ATT
# RUN: llvm-objdump -dMintel %t | FileCheck %s --check-prefix=INTEL
# RUN: llvm-objdump -d --disassembler-options=intel %t | FileCheck %s --check-prefix=INTEL

## The last wins.
# RUN: llvm-objdump -dM att -M att,intel %t | FileCheck %s --check-prefix=INTEL

## Test discouraged internal cl::opt options.
# RUN: llvm-objdump -d --x86-asm-syntax=att %t | FileCheck %s --check-prefix=ATT
# RUN: llvm-objdump -d --x86-asm-syntax=intel %t | FileCheck %s --check-prefix=INTEL

# ATT: movw $1, %ax
# ATT: imull %esi, %edi
# ATT: leaq 5(%rsi,%rdi,4), %rax

# INTEL: mov ax, 1
# INTEL: imul edi, esi
# INTEL: lea rax, [rsi + 4*rdi + 5]

  movw $1, %ax
  imull %esi, %edi
  leaq 5(%rsi,%rdi,4), %rax
