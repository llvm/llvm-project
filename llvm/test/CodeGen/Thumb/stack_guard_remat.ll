; RUN: llc < %s -mtriple=thumb-apple-darwin -relocation-model=pic -no-integrated-as | FileCheck %s -check-prefix=PIC
; RUN: llc < %s -mtriple=thumb-apple-darwin -relocation-model=static -no-integrated-as | FileCheck %s -check-prefix=NO-PIC  -check-prefix=STATIC
; RUN: llc < %s -mtriple=thumb-apple-darwin -relocation-model=dynamic-no-pic -no-integrated-as | FileCheck %s  -check-prefix=NO-PIC -check-prefix=DYNAMIC-NO-PIC

;PIC:        foo2
;PIC:        add [[SAVED_GUARD:r[0-9]+]], sp, #904
;PIC-NEXT:   ldr [[SAVED_GUARD]], [[[SAVED_GUARD]], #124]
;PIC-NEXT:   ldr [[ORIGINAL_GUARD:r[0-9]+]], [[ORIGINAL_GUARD_LABEL:LCPI[0-9_]+]]
;PIC-NEXT: [[LABEL1:LPC[0-9_]+]]:
;PIC-NEXT:   add [[ORIGINAL_GUARD]], pc
;PIC-NEXT:   ldr [[ORIGINAL_GUARD]], [[[ORIGINAL_GUARD]]]
;PIC-NEXT:   ldr [[ORIGINAL_GUARD]], [[[ORIGINAL_GUARD]]]
;PIC-NEXT:   cmp [[ORIGINAL_GUARD]], [[SAVED_GUARD]]

;PIC:      [[ORIGINAL_GUARD_LABEL]]:
;PIC-NEXT:   .long L___stack_chk_guard$non_lazy_ptr-([[LABEL1]]+4)

;NO-PIC:   foo2
;NO-PIC:                add [[SAVED_GUARD:r[0-9]+]], sp, #904
;NO-PIC-NEXT:           ldr [[SAVED_GUARD]], [[[SAVED_GUARD]], #124]
;NO-PIC-NEXT:           ldr [[ORIGINAL_GUARD:r[0-9]+]], [[ORIGINAL_GUARD_LABEL:LCPI[0-9_]+]]
;NO-PIC-NOT: LPC
;NO-PIC-NEXT:           ldr [[ORIGINAL_GUARD]], [[[ORIGINAL_GUARD]]]
;DYNAMIC-NO-PIC-NEXT:   ldr [[ORIGINAL_GUARD]], [[[ORIGINAL_GUARD]]]
;NO-PIC-NEXT:           cmp [[ORIGINAL_GUARD]], [[SAVED_GUARD]]

;STATIC:      [[ORIGINAL_GUARD_LABEL]]:
;STATIC-NEXT:   .long ___stack_chk_guard

;DYNAMIC-NO-PIC:      [[ORIGINAL_GUARD_LABEL]]:
;DYNAMIC-NO-PIC-NEXT:   .long L___stack_chk_guard$non_lazy_ptr

; Function Attrs: nounwind ssp
define i32 @test_stack_guard_remat() #0 {
  %a1 = alloca [256 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 1024, ptr %a1)
  call void @foo3(ptr %a1) #3
  call void asm sideeffect "foo2", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{sp},~{lr}"()
  call void @llvm.lifetime.end.p0(i64 1024, ptr %a1)
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

declare void @foo3(ptr)

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"PIC Level", i32 2}
