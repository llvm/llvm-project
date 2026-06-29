# REQUIRES: asserts
## Verify that boundary alignment converges in O(1) inner iterations,
## not O(N) where N is the number of BoundaryAlign fragments.
## The fused relaxation+layout pass gives each BoundaryAlign fragment fresh
## upstream offsets, so all padding is computed correctly in a single pass.

# RUN: llvm-mc -filetype=obj -triple=x86_64 --stats \
# RUN:   --x86-align-branch-boundary=32 --x86-align-branch=jcc+call %s \
# RUN:   -o %t 2>&1 | FileCheck %s --check-prefix=STATS
# STATS: 3 assembler - Number of assembler layout and relaxation steps

# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# CHECK:        0: testl
# CHECK:        2: je
# CHECK:        8: callq
# CHECK:       1c: nop
# CHECK-NEXT:  20: callq
# CHECK:       34: testl
# CHECK-NEXT:  36: jne
# CHECK:       65: testl
# CHECK:       67: je
# CHECK:       a0: jne
# CHECK:       d1: je
# CHECK:      100: callq
# CHECK:      105: testl
# CHECK-NEXT: 107: jne
# CHECK:      13b: je
# CHECK:      160: callq
# CHECK:      16a: testl
# CHECK-NEXT: 16c: jne
# CHECK:      172: retq

  .p2align 5
func:
  testl %eax, %eax
  je .Lend
  .rept 8
  callq foo
  .endr
  testl %ecx, %ecx
  jne func
  .rept 8
  callq foo
  .endr
  testl %edx, %edx
  je .Lend
  .rept 8
  callq foo
  .endr
  testl %esi, %esi
  jne func
  .rept 8
  callq foo
  .endr
  testl %edi, %edi
  je .Lend
  .rept 8
  callq foo
  .endr
  testl %ebp, %ebp
  jne func
  .rept 8
  callq foo
  .endr
  testl %eax, %eax
  je .Lend
  .rept 8
  callq foo
  .endr
  testl %ecx, %ecx
  jne func
.Lend:
  retq
