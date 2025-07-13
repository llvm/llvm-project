; RUN: llc -relocation-model=static -mattr=+no-movt < %s | FileCheck %s

target triple = "thumbv7a-linux-gnueabi"

define i32 @test1() #0 {
; CHECK-LABEL: test1:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    push	{r7, lr}
; CHECK-NEXT:    sub.w	sp, sp, #1032
; CHECK-NEXT:    ldr	r0, .LCPI0_0
; CHECK-NEXT:    ldr	r0, [r0]
; CHECK-NEXT:    str.w	r0, [sp, #1028]
; CHECK-NEXT:    add	r0, sp, #4
; CHECK-NEXT:    bl	foo
; CHECK-NEXT:    ldr.w	r0, [sp, #1028]
; CHECK-NEXT:    ldr	r1, .LCPI0_0
; CHECK-NEXT:    ldr	r1, [r1]
; CHECK-NEXT:    cmp	r1, r0
; CHECK-NEXT:    ittt	eq
; CHECK-NEXT:    moveq	r0, #0
; CHECK-NEXT:    addeq.w	sp, sp, #1032
; CHECK-NEXT:    popeq	{r7, pc}
; CHECK-NEXT:  .LBB0_1:
; CHECK-NEXT:    bl __stack_chk_fail
  %a1 = alloca [256 x i32], align 4
  call void @foo(ptr %a1) #3
  ret i32 0
}

declare void @foo(ptr)

attributes #0 = { nounwind sspstrong }
