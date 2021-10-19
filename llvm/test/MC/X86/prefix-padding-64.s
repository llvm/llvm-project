# RUN: llvm-mc -mcpu=skylake -filetype=obj -triple x86_64-pc-linux-gnu %s -x86-pad-max-prefix-size=15 -x86-pad-for-align=1 | llvm-objdump -d - | FileCheck %s

# Check prefix padding generation for all cases on 64 bit x86.

# CHECK:  0: 2e 2e 2e 2e 2e 2e 2e 2e 48 81 e1 00 00 00 00   andq  $0, %rcx
# CHECK:  f: 2e 2e 2e 2e 2e 2e 2e 2e 48 81 21 00 00 00 00   andq  $0, %cs:(%rcx)
# CHECK: 1e: 2e 2e 2e 2e 2e 2e 2e 2e 48 81 21 00 00 00 00   andq  $0, %cs:(%rcx)
# CHECK: 2d: 3e 3e 3e 3e 3e 3e 3e 3e 48 81 21 00 00 00 00   andq  $0, %ds:(%rcx)
# CHECK: 3c: 26 26 26 26 26 26 26 26 48 81 21 00 00 00 00   andq  $0, %es:(%rcx)
# CHECK: 4b: 64 64 64 64 64 64 64 64 48 81 21 00 00 00 00   andq  $0, %fs:(%rcx)
# CHECK: 5a: 65 65 65 65 65 65 65 65 48 81 21 00 00 00 00   andq  $0, %gs:(%rcx)
# CHECK: 69: 36 36 36 36 36 36 36 36 48 81 21 00 00 00 00   andq  $0, %ss:(%rcx)
# CHECK: 78: 2e 2e 2e 2e 48 81 a1 00 00 00 00 00 00 00 00   andq  $0, %cs:(%rcx)
# CHECK: 87: 2e 2e 2e 2e 48 81 a1 00 00 00 00 00 00 00 00   andq  $0, %cs:(%rcx)
# CHECK: 96: 2e 2e 2e 2e 2e 2e 2e 48 81 24 24 00 00 00 00   andq  $0, %cs:(%rsp)
# CHECK: a5: 65 65 65 65 65 65 65 48 81 24 24 00 00 00 00   andq  $0, %gs:(%rsp)
# CHECK: b4: 2e 2e 2e 48 81 a4 24 00 00 00 00 00 00 00 00   andq  $0, %cs:(%rsp)
# CHECK: c3: 2e 2e 2e 2e 2e 2e 2e 48 81 65 00 00 00 00 00   andq  $0, %cs:(%rbp)
# CHECK: d2: 65 65 65 65 65 65 65 48 81 65 00 00 00 00 00   andq  $0, %gs:(%rbp)
# CHECK: e1: 2e 2e 2e 2e 48 81 a5 00 00 00 00 00 00 00 00   andq  $0, %cs:(%rbp)
  .text
  .section  .text
  .p2align 8
  # non-memory
  andq $foo, %rcx
  # memory, non-esp/ebp
  andq $foo, (%rcx)
  andq $foo, %cs:(%rcx)
  andq $foo, %ds:(%rcx)
  andq $foo, %es:(%rcx)
  andq $foo, %fs:(%rcx)
  andq $foo, %gs:(%rcx)
  andq $foo, %ss:(%rcx)
  andq $foo, data16 (%rcx)
  andq $foo, data32 (%rcx)
  # esp w/o segment override
  andq $foo, (%rsp)
  andq $foo, %gs:(%rsp)
  andq $foo, data32 (%rsp)
  # ebp w/o segment override
  andq $foo, (%rbp)
  andq $foo, %gs:(%rbp)
  andq $foo, data32 (%rbp)

  # Request enough padding to justify padding all of the above
  .p2align 8
  int3

  .section "other"
bar:
  .p2align 3
  int3  
foo:
