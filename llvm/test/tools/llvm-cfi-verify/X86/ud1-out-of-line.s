# RUN: llvm-mc %s -filetype obj -triple x86_64-linux-elf -o %t.o
# RUN: llvm-cfi-verify --ignore-dwarf %t.o | FileCheck %s

# Verify that out-of-line CFI checks branching to UBSan ud1l traps are
# classified as PROTECTED across fallthrough and split-block layouts.

# CHECK: {{^Instruction: .* \(PROTECTED\)}}
# CHECK: {{^Instruction: .* \(PROTECTED\)}}
# CHECK-NOT: UNPROTECTED

# CHECK: Total Indirect CF Instructions: 2
# CHECK: Expected Protected: 2 (100.00%)
# CHECK: Unexpected Protected: 0 (0.00%)
# CHECK: Expected Unprotected: 0 (0.00%)
# CHECK: Unexpected Unprotected (BAD): 0 (0.00%)

  .text
  .globl f_call
  .p2align 4, 0x90
  .type f_call,@function
f_call:
  movl $0xff439eb2, %r10d
  addl -4(%rdi), %r10d
  jne .Ltrap1
  callq *%rdi
  retq
.Ltrap1:
  ud1l 27(%eax), %eax
.Lfunc_end0:
  .size f_call, .Lfunc_end0-f_call

  .globl f_split_tailcall
  .p2align 4, 0x90
  .type f_split_tailcall,@function
f_split_tailcall:
  movl $0xff439eb2, %r10d
  addl -4(%rdi), %r10d
  jne .Ltrap2
  jmp .Lcall2
.Ltrap2:
  ud1l 27(%eax), %eax
.Lcall2:
  jmpq *%rdi
.Lfunc_end1:
  .size f_split_tailcall, .Lfunc_end1-f_split_tailcall
