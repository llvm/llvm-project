; RUN: llc < %s -mtriple=thumbv7-apple-ios -relocation-model=pic -no-integrated-as | FileCheck %s -check-prefix=PIC
; RUN: llc < %s -mtriple=thumbv7-apple-ios -relocation-model=static -no-integrated-as | FileCheck %s -check-prefix=STATIC
; RUN: llc < %s -mtriple=thumbv7-apple-ios -relocation-model=dynamic-no-pic -no-integrated-as | FileCheck %s  -check-prefix=DYNAMIC-NO-PIC

;PIC:   foo2
;PIC:   movw  [[R0:r[0-9]+]], :lower16:(L___stack_chk_guard$non_lazy_ptr-([[LABEL0:LPC[0-9_]+]]+4))
;PIC:   movt  [[R0]], :upper16:(L___stack_chk_guard$non_lazy_ptr-([[LABEL0]]+4))
;PIC: [[LABEL0]]:
;PIC:   add [[R0]], pc
;PIC:   ldr [[R1:r[0-9]+]], {{\[}}[[R0]]{{\]}}
;PIC:   ldr {{r[0-9]+}}, {{\[}}[[R1]]{{\]}}

;STATIC:   foo2
;STATIC:   movw  [[R0:r[0-9]+]], :lower16:___stack_chk_guard
;STATIC:   movt  [[R0]], :upper16:___stack_chk_guard
;STATIC:   ldr {{r[0-9]+}}, {{\[}}[[R0]]{{\]}}

;DYNAMIC-NO-PIC:   foo2
;DYNAMIC-NO-PIC:   movw  [[R0:r[0-9]+]], :lower16:L___stack_chk_guard$non_lazy_ptr
;DYNAMIC-NO-PIC:   movt  [[R0]], :upper16:L___stack_chk_guard$non_lazy_ptr
;DYNAMIC-NO-PIC:   ldr {{r[0-9]+}}, {{\[}}[[R0]]{{\]}}

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

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
