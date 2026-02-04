; RUN: llc -mtriple thumbv7em-apple-darwin -o - < %s | FileCheck %s

%"struct.s1" = type { [19 x i32] }

define void @f0(ptr byval(%"struct.s1") %0, ptr %1) #1 {
; CHECK-LABEL: _f0:                                    @ @f0
; CHECK-NEXT:  @ %bb.0:
; CHECK-NEXT:  	sub	sp, #16
; CHECK-NEXT:  	push	{r4, lr}
; CHECK-NEXT:  	sub	sp, #76
; CHECK-NEXT:  	add.w	r9, sp, #84
; CHECK-NEXT:  	stm.w	r9, {r0, r1, r2, r3}
; CHECK-NEXT:  	mov	r0, sp
; CHECK-NEXT:  	add	r1, sp, #84
; CHECK-NEXT:  	movs	r2, #76
; CHECK-NEXT:  	mov	r3, r0
; CHECK-NEXT:  LBB0_1:                                 @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:  	ldr	r4, [r1], #4
; CHECK-NEXT:  	subs	r2, #4
; CHECK-NEXT:  	str	r4, [r3], #4
; CHECK-NEXT:  	bne	LBB0_1
; CHECK-NEXT:  @ %bb.2:
; CHECK-NEXT:  	add.w	r1, r0, #12
; CHECK-NEXT:  	add	r2, sp, #100
; CHECK-NEXT:  	ldr	r0, [sp, #160]
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldr	r3, [r1], #4
; CHECK-NEXT:  	str	r3, [r2], #4
; CHECK-NEXT:  	ldm.w	sp, {r1, r2, r3}
; CHECK-NEXT:  	add	sp, #76
; CHECK-NEXT:  	pop.w	{r4, lr}
; CHECK-NEXT:  	add	sp, #16
; CHECK-NEXT:  	b.w	_f1
  tail call  void @f1(ptr %1, ptr byval(%"struct.s1") %0)
  ret void
}

declare void @f1(ptr, ptr)

attributes #1 = { nounwind "frame-pointes"="non-leaf" }
