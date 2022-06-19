; RUN: llc < %s -mtriple=thumb-apple-darwin -relocation-model=pic -no-integrated-as | FileCheck %s -check-prefix=PIC
; RUN: llc < %s -mtriple=thumb-apple-darwin -relocation-model=static -no-integrated-as | FileCheck %s -check-prefix=NO-PIC  -check-prefix=STATIC
; RUN: llc < %s -mtriple=thumb-apple-darwin -relocation-model=dynamic-no-pic -no-integrated-as | FileCheck %s  -check-prefix=NO-PIC -check-prefix=DYNAMIC-NO-PIC

;PIC:        foo2
;PIC:        ldr [[SAVED_GUARD:r[0-9]+]], [[GUARD_STACK_OFFSET:LCPI[0-9_]+]]
;PIC-NEXT:   add [[SAVED_GUARD]], sp
;PIC-NEXT:   ldr [[SAVED_GUARD]], [[[SAVED_GUARD]]]
;PIC-NEXT:   ldr [[ORIGINAL_GUARD:r[0-9]+]], [[ORIGINAL_GUARD_LABEL:LCPI[0-9_]+]]
;PIC-NEXT: [[LABEL1:LPC[0-9_]+]]:
;PIC-NEXT:   add [[ORIGINAL_GUARD]], pc
;PIC-NEXT:   ldr [[ORIGINAL_GUARD]], [[[ORIGINAL_GUARD]]]
;PIC-NEXT:   ldr [[ORIGINAL_GUARD]], [[[ORIGINAL_GUARD]]]
;PIC-NEXT:   cmp [[ORIGINAL_GUARD]], [[SAVED_GUARD]]

;PIC:      [[GUARD_STACK_OFFSET]]:
;PIC-NEXT:   .long 1028
;PIC:      [[ORIGINAL_GUARD_LABEL]]:
;PIC-NEXT:   .long L___stack_chk_guard$non_lazy_ptr-([[LABEL1]]+4)

;NO-PIC:   foo2
;NO-PIC:                ldr [[SAVED_GUARD:r[0-9]+]], [[GUARD_STACK_OFFSET:LCPI[0-9_]+]]
;NO-PIC-NEXT:           add [[SAVED_GUARD]], sp
;NO-PIC-NEXT:           ldr [[SAVED_GUARD]], [[[SAVED_GUARD]]]
;NO-PIC-NEXT:           ldr [[ORIGINAL_GUARD:r[0-9]+]], [[ORIGINAL_GUARD_LABEL:LCPI[0-9_]+]]
;NO-PIC-NOT: LPC
;NO-PIC-NEXT:           ldr [[ORIGINAL_GUARD]], [[[ORIGINAL_GUARD]]]
;DYNAMIC-NO-PIC-NEXT:   ldr [[ORIGINAL_GUARD]], [[[ORIGINAL_GUARD]]]
;NO-PIC-NEXT:           cmp [[ORIGINAL_GUARD]], [[SAVED_GUARD]]

;STATIC:      [[GUARD_STACK_OFFSET]]:
;STATIC-NEXT:   .long 1028
;STATIC:      [[ORIGINAL_GUARD_LABEL]]:
;STATIC-NEXT:   .long ___stack_chk_guard

;DYNAMIC-NO-PIC:      [[GUARD_STACK_OFFSET]]:
;DYNAMIC-NO-PIC-NEXT:   .long 1028
;DYNAMIC-NO-PIC:      [[ORIGINAL_GUARD_LABEL]]:
;DYNAMIC-NO-PIC-NEXT:   .long L___stack_chk_guard$non_lazy_ptr

; Function Attrs: nounwind ssp
define i32 @test_stack_guard_remat() #0 {
  %a1 = alloca [256 x i32], align 4
  %1 = bitcast [256 x i32]* %a1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1024, i8* %1)
  %2 = getelementptr inbounds [256 x i32], [256 x i32]* %a1, i32 0, i32 0
  call void @foo3(i32* %2) #3
  call void asm sideeffect "foo2", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{sp},~{lr}"()
  call void @llvm.lifetime.end.p0i8(i64 1024, i8* %1)
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare void @foo3(i32*)

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
