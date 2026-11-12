; RUN: llc -mtriple armv6t2 %s -o - | FileCheck %s
; RUN: llc -mtriple thumbv6t2 %s -o - | FileCheck %s --check-prefix=CHECK-T2
; RUN: llc -mtriple armv7 %s -o - | FileCheck %s
; RUN: llc -mtriple thumbv7 %s -o - | FileCheck %s --check-prefix=CHECK-T2
; RUN: llc -mtriple thumbv7m %s -o - | FileCheck %s --check-prefix=CHECK-T2
; RUN: llc -mtriple thumbv8m.main %s -o - | FileCheck %s --check-prefix=CHECK-T2

; CHECK-LABEL: unfold1
; CHECK-NOT: mov
; CHECK: orr r0, r0, #255
; CHECK: add r0, r1, r0, lsl #1
; CHECK-T2-NOT: mov
; CHECK-T2: orr r0, r0, #255
; CHECK-T2: add.w r0, r1, r0, lsl #1
define arm_aapcscc i32 @unfold1(i32 %a, i32 %b) {
entry:
  %or = shl i32 %a, 1
  %shl = or i32 %or, 510
  %add = add nsw i32 %shl, %b
  ret i32 %add
}

; CHECK-LABEL: unfold2
; CHECK-NOT: mov
; CHECK: orr r0, r0, #4080
; CHECK: sub r0, r1, r0, lsl #2
; CHECK-T2-NOT: mov
; CHECK-T2: orr r0, r0, #4080
; CHECK-T2: sub.w r0, r1, r0, lsl #2
define arm_aapcscc i32 @unfold2(i32 %a, i32 %b) {
entry:
  %or = shl i32 %a, 2
  %shl = or i32 %or, 16320
  %sub = sub nsw i32 %b, %shl
  ret i32 %sub
}

; CHECK-LABEL: unfold3
; CHECK-NOT: mov
; CHECK: orr r0, r0, #65280
; CHECK: and r0, r1, r0, lsl #4
; CHECK-T2-NOT: mov
; CHECK-T2: orr r0, r0, #65280
; CHECK-T2: and.w r0, r1, r0, lsl #4
define arm_aapcscc i32 @unfold3(i32 %a, i32 %b) {
entry:
  %or = shl i32 %a, 4
  %shl = or i32 %or, 1044480
  %and = and i32 %shl, %b
  ret i32 %and
}

; CHECK-LABEL: unfold4
; CHECK-NOT: mov
; CHECK: orr r0, r0, #1044480
; CHECK: eor r0, r1, r0, lsl #5
; CHECK-T2-NOT: mov
; CHECK-T2: orr r0, r0, #1044480
; CHECK-T2: eor.w r0, r1, r0, lsl #5
define arm_aapcscc i32 @unfold4(i32 %a, i32 %b) {
entry:
  %or = shl i32 %a, 5
  %shl = or i32 %or, 33423360
  %xor = xor i32 %shl, %b
  ret i32 %xor
}

; CHECK-LABEL: unfold5
; CHECK-NOT: mov
; CHECK: add r0, r0, #496
; CHECK: orr r0, r1, r0, lsl #6
; CHECK-T2: add.w r0, r0, #496
; CHECK-T2: orr.w r0, r1, r0, lsl #6
define arm_aapcscc i32 @unfold5(i32 %a, i32 %b) {
entry:
  %add = shl i32 %a, 6
  %shl = add i32 %add, 31744
  %or = or i32 %shl, %b
  ret i32 %or
}

; CHECK-LABEL: unfold6
; CHECK-NOT: mov
; CHECK: add r0, r0, #7936
; CHECK: and r0, r1, r0, lsl #8
; CHECK-T2-NOT: mov
; CHECK-T2: add.w r0, r0, #7936
; CHECK-T2: and.w r0, r1, r0, lsl #8
define arm_aapcscc i32 @unfold6(i32 %a, i32 %b) {
entry:
  %add = shl i32 %a, 8
  %shl = add i32 %add, 2031616
  %and = and i32 %shl, %b
  ret i32 %and
}

; CHECK-LABEL: unfold7
; CHECK-NOT: mov
; CHECK: and r0, r0, #256
; CHECK: add r0, r1, r0, lsl #1
; CHECK-T2-NOT: mov
; CHECK-T2: and r0, r0, #256
; CHECK-T2: add.w r0, r1, r0, lsl #1
define arm_aapcscc i32 @unfold7(i32 %a, i32 %b) {
entry:
  %shl = shl i32 %a, 1
  %and = and i32 %shl, 512
  %add = add nsw i32 %and, %b
  ret i32 %add
}

; CHECK-LABEL: unfold8
; CHECK-NOT: mov
; CHECK: add r0, r0, #126976
; CHECK: eor r0, r1, r0, lsl #9
; CHECK-T2-NOT: mov
; CHECK-T2: add.w r0, r0, #126976
; CHECK-T2: eor.w r0, r1, r0, lsl #9
define arm_aapcscc i32 @unfold8(i32 %a, i32 %b) {
entry:
  %add = shl i32 %a, 9
  %shl = add i32 %add, 65011712
  %xor = xor i32 %shl, %b
  ret i32 %xor
}

; CHECK-LABEL: unfold9
; CHECK-NOT: mov
; CHECK: eor r0, r0, #255
; CHECK: add r0, r1, r0, lsl #1
; CHECK-T2-NOT: mov
; CHECK-T2: eor r0, r0, #255
; CHECK-T2: add.w r0, r1, r0, lsl #1
define arm_aapcscc i32 @unfold9(i32 %a, i32 %b) {
entry:
  %shl = shl i32 %a, 1
  %xor = xor i32 %shl, 510
  %add = add nsw i32 %xor, %b
  ret i32 %add
}

; CHECK-LABEL: unfold10
; CHECK-NOT: mov r2
; CHECK: orr r2, r0, #4080
; CHECK: cmp r1, r2, lsl #10
; CHECK-T2-NOT: mov.w r2
; CHECK-T2: orr r2, r0, #4080
; CHECK-T2: cmp.w r1, r2, lsl #10
define arm_aapcscc i32 @unfold10(i32 %a, i32 %b) {
entry:
  %or = shl i32 %a, 10
  %shl = or i32 %or, 4177920
  %cmp = icmp sgt i32 %shl, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: unfold11
; CHECK-NOT: mov r2
; CHECK: add r2, r0, #7936
; CHECK: cmp r1, r2, lsl #11
; CHECK-T2-NOT: mov.w r2
; CHECK-T2: add.w r2, r0, #7936
; CHECK-T2: cmp.w r1, r2, lsl #11
define arm_aapcscc i32 @unfold11(i32 %a, i32 %b) {
entry:
  %add = shl i32 %a, 11
  %shl = add i32 %add, 16252928
  %cmp = icmp sgt i32 %shl, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

