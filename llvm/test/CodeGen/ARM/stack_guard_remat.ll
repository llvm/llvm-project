; RUN: llc < %s -mtriple=arm-apple-ios -relocation-model=pic -no-integrated-as | FileCheck %s -check-prefix=PIC
; RUN: llc < %s -mtriple=arm-apple-ios -relocation-model=static -no-integrated-as | FileCheck %s -check-prefix=NO-PIC -check-prefix=STATIC
; RUN: llc < %s -mtriple=arm-apple-ios -relocation-model=dynamic-no-pic -no-integrated-as | FileCheck %s  -check-prefix=NO-PIC -check-prefix=DYNAMIC-NO-PIC
; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=cortex-a8 -relocation-model=pic -no-integrated-as | FileCheck %s -check-prefix=PIC-V7
; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=cortex-a8 -relocation-model=static -no-integrated-as | FileCheck %s -check-prefix=STATIC-V7
; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=cortex-a8 -relocation-model=dynamic-no-pic -no-integrated-as | FileCheck %s -check-prefix=DYNAMIC-NO-PIC-V7

;PIC:   foo2
;PIC:   ldr [[R0:r[0-9]+]], [[LABEL0:LCPI[0-9_]+]]
;PIC: [[LABEL1:LPC0_1]]:
;PIC:   add [[R1:r[0-9]+]], pc, [[R0]]
;PIC:   ldr [[R2:r[0-9]+]], [[[R1]]]
;PIC:   ldr {{r[0-9]+}}, [[[R2]]]

;PIC:      [[LABEL0]]:
;PIC-NEXT:   .long L___stack_chk_guard$non_lazy_ptr-([[LABEL1]]+8)

;NO-PIC: foo2
;NO-PIC: ldr [[R0:r[0-9]+]], [[LABEL0:LCPI[0-9_]+]]
;NO-PIC-NOT: LPC
;NO-PIC: ldr {{r[0-9]+}}, [[[R0]]]

;STATIC:      [[LABEL0]]:
;STATIC-NEXT:   .long ___stack_chk_guard

;DYNAMIC-NO-PIC:      [[LABEL0]]:
;DYNAMIC-NO-PIC-NEXT:   .long L___stack_chk_guard$non_lazy_ptr

;PIC-V7:   movw [[R0:r[0-9]+]], :lower16:(L___stack_chk_guard$non_lazy_ptr-([[LABEL0:LPC[0-9_]+]]+8))
;PIC-V7:   movt [[R0]], :upper16:(L___stack_chk_guard$non_lazy_ptr-([[LABEL0]]+8))
;PIC-V7: [[LABEL0]]:
;PIC-V7:   ldr [[R0]], [pc, [[R0]]]
;PIC-V7:   ldr [[R0]], [[[R0]]]

;PIC-V7: L___stack_chk_guard$non_lazy_ptr:
;PIC-V7:   .indirect_symbol        ___stack_chk_guard

;STATIC-V7: movw [[R0:r[0-9]+]], :lower16:___stack_chk_guard
;STATIC-V7: movt [[R0]], :upper16:___stack_chk_guard
;STATIC-V7: ldr  [[R0]], [[[R0]]]

;DYNAMIC-NO-PIC-V7: movw [[R0:r[0-9]+]], :lower16:L___stack_chk_guard$non_lazy_ptr
;DYNAMIC-NO-PIC-V7: movt [[R0]], :upper16:L___stack_chk_guard$non_lazy_ptr
;DYNAMIC-NO-PIC-V7: ldr  [[R0]], [[[R0]]]
;DYNAMIC-NO-PIC-V7: ldr  [[R0]], [[[R0]]]

;DYNAMIC-NO-PIC-V7: L___stack_chk_guard$non_lazy_ptr:
;DYNAMIC-NO-PIC-V7:   .indirect_symbol        ___stack_chk_guard

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
