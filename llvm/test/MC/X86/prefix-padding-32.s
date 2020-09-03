# RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -x86-pad-max-prefix-size=15 | llvm-objdump -d --section=.text - | FileCheck %s

# Check prefix padding generation for all cases on 32 bit x86.

# CHECK:  1: 3e 3e 3e 3e 3e 3e 3e 3e 3e 81 e1 01 00 00 00 	andl	$1, %ecx
# CHECK: 10: 3e 3e 3e 3e 3e 3e 3e 3e 3e 81 21 01 00 00 00 	andl	$1, %ds:(%ecx)
# CHECK: 1f: 2e 2e 2e 2e 2e 2e 2e 2e 2e 81 21 01 00 00 00 	andl	$1, %cs:(%ecx)
# CHECK: 2e: 3e 3e 3e 3e 3e 3e 3e 3e 3e 81 21 01 00 00 00 	andl	$1, %ds:(%ecx)
# CHECK: 3d: 26 26 26 26 26 26 26 26 26 81 21 01 00 00 00 	andl	$1, %es:(%ecx)
# CHECK: 4c: 64 64 64 64 64 64 64 64 64 81 21 01 00 00 00 	andl	$1, %fs:(%ecx)
# CHECK: 5b: 65 65 65 65 65 65 65 65 65 81 21 01 00 00 00 	andl	$1, %gs:(%ecx)
# CHECK: 6a: 36 36 36 36 36 36 36 36 36 81 21 01 00 00 00 	andl	$1, %ss:(%ecx)
# CHECK: 79: 3e 3e 3e 3e 3e 81 a1 00 00 00 00 01 00 00 00 	andl	$1, %ds:(%ecx)
# CHECK: 88: 3e 3e 3e 3e 3e 81 a1 00 00 00 00 01 00 00 00 	andl	$1, %ds:(%ecx)
# CHECK: 97: 36 36 36 36 36 36 36 36 81 24 24 01 00 00 00 	andl	$1, %ss:(%esp)
# CHECK: a6: 65 65 65 65 65 65 65 65 81 24 24 01 00 00 00 	andl	$1, %gs:(%esp)
# CHECK: b5: 36 36 36 36 81 a4 24 00 00 00 00 01 00 00 00 	andl	$1, %ss:(%esp)
# CHECK: c4: 36 36 36 36 36 36 36 36 81 65 00 01 00 00 00 	andl	$1, %ss:(%ebp)
# CHECK: d3: 65 65 65 65 65 65 65 65 81 65 00 01 00 00 00 	andl	$1, %gs:(%ebp)
# CHECK: e2: 36 36 36 36 36 81 a5 00 00 00 00 01 00 00 00 	andl	$1, %ss:(%ebp)
  .text
  .section  .text
  .p2align 8
bar:
  int3  
foo:
  # non-memory
  andl $foo, %ecx
  # memory, non-esp/ebp
  andl $foo, (%ecx)
  andl $foo, %cs:(%ecx)
  andl $foo, %ds:(%ecx)
  andl $foo, %es:(%ecx)
  andl $foo, %fs:(%ecx)
  andl $foo, %gs:(%ecx)
  andl $foo, %ss:(%ecx)
  andl $foo, data16 (%ecx)
  andl $foo, data32 (%ecx)
  # esp w/o segment override
  andl $foo, (%esp)
  andl $foo, %gs:(%esp)
  andl $foo, data32 (%esp)
  # ebp w/o segment override
  andl $foo, (%ebp)
  andl $foo, %gs:(%ebp)
  andl $foo, data32 (%ebp)

  # Request enough padding to justify padding all of the above
  .p2align 8
  int3
