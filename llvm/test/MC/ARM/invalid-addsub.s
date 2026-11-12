@ RUN: not llvm-mc -triple thumbv7-apple-ios %s -o /dev/null 2>&1 | FileCheck %s
add sp, r5, #1
addw sp, r7, #4
add sp, r3, r2
add sp, r3, r5, lsl #3
sub sp, r5, #1
subw sp, r7, #4
sub sp, r3, r2
sub sp, r3, r5, lsl #3
@CHECK: error: invalid instruction, any one of the following would fix this:
@CHECK-NEXT: add sp, r5, #1
@CHECK-NEXT: ^
@CHECK-NEXT: note: invalid operand for instruction
@CHECK-NEXT: add sp, r5, #1
@CHECK-NEXT:             ^
@CHECK-NEXT: note: operand must be a register in range [r0, r12] or r14
@CHECK-NEXT: add sp, r5, #1
@CHECK-NEXT:             ^
@CHECK-NEXT: note: operand must be a register in range [r0, r12] or r14
@CHECK-NEXT: add sp, r5, #1
@CHECK-NEXT:     ^
@CHECK-NEXT: note: operand must be a register sp
@CHECK-NEXT: add sp, r5, #1
@CHECK-NEXT:         ^
@CHECK-NEXT: error: invalid instruction, any one of the following would fix this:
@CHECK-NEXT: addw sp, r7, #4
@CHECK-NEXT: ^
@CHECK-NEXT: note: operand must be a register in range [r0, r12] or r14
@CHECK-NEXT: addw sp, r7, #4
@CHECK-NEXT:      ^
@CHECK-NEXT: note: operand must be a register sp
@CHECK-NEXT: addw sp, r7, #4
@CHECK-NEXT:          ^
@CHECK-NEXT: error: source register must be sp if destination is sp
@CHECK-NEXT: add sp, r3, r2
@CHECK-NEXT:         ^
@CHECK-NEXT: error: source register must be sp if destination is sp
@CHECK-NEXT: add sp, r3, r5, lsl #3
@CHECK-NEXT:         ^
@CHECK-NEXT: error: invalid instruction, any one of the following would fix this:
@CHECK-NEXT: sub sp, r5, #1
@CHECK-NEXT: ^
@CHECK-NEXT: note: invalid operand for instruction
@CHECK-NEXT: sub sp, r5, #1
@CHECK-NEXT:             ^
@CHECK-NEXT: note: operand must be a register in range [r0, r12] or r14
@CHECK-NEXT: sub sp, r5, #1
@CHECK-NEXT:             ^
@CHECK-NEXT: note: operand must be a register in range [r0, r12] or r14
@CHECK-NEXT: sub sp, r5, #1
@CHECK-NEXT:     ^
@CHECK-NEXT: note: operand must be a register sp
@CHECK-NEXT: sub sp, r5, #1
@CHECK-NEXT:         ^
@CHECK-NEXT: error: invalid instruction, any one of the following would fix this:
@CHECK-NEXT: subw sp, r7, #4
@CHECK-NEXT: ^
@CHECK-NEXT: note: operand must be a register in range [r0, r12] or r14
@CHECK-NEXT: subw sp, r7, #4
@CHECK-NEXT:      ^
@CHECK-NEXT: note: operand must be a register sp
@CHECK-NEXT: subw sp, r7, #4
@CHECK-NEXT:          ^
@CHECK-NEXT: error: source register must be sp if destination is sp
@CHECK-NEXT: sub sp, r3, r2
@CHECK-NEXT:         ^
@CHECK-NEXT: error: source register must be sp if destination is sp
@CHECK-NEXT: sub sp, r3, r5, lsl #3
