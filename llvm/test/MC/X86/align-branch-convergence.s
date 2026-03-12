# REQUIRES: asserts
## Verify that boundary alignment converges in O(1) inner iterations,
## not O(N) where N is the number of BoundaryAlign fragments.
## The fused relaxation+layout pass gives each BoundaryAlign fragment fresh
## upstream offsets, so all padding is computed correctly in a single pass.

# RUN: llvm-mc -filetype=obj -triple=x86_64 --stats \
# RUN:   --x86-align-branch-boundary=32 --x86-align-branch=jcc+call %s \
# RUN:   -o %t 2>&1 | FileCheck %s --check-prefix=STATS
# STATS: 2 assembler - Number of assembler layout and relaxation steps

# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# CHECK:       0: testl
# CHECK:       2: je
# CHECK:       4: callq
# CHECK:      1d: nop
# CHECK-NEXT: 20: callq
# CHECK:      2f: testl
# CHECK-NEXT: 31: jne
# CHECK:      3d: nop
# CHECK-NEXT: 40: callq
# CHECK:      4a: retq

  .p2align 5
func:
  testl %eax, %eax
  je .Lend
  .rept 8
  callq foo
  .endr
  testl %ecx, %ecx
  jne func
  .rept 4
  callq foo
  .endr
.Lend:
  retq
