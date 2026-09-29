; RUN: llc -mtriple thumbv7-windows -mcpu cortex-a9 -o - %s \
; RUN:     | FileCheck %s -check-prefix CHECK-DEFAULT-CODE-MODEL

; RUN: llc -mtriple thumbv7-windows -mcpu cortex-a9 -code-model large -o - %s \
; RUN:     | FileCheck %s -check-prefix CHECK-LARGE-CODE-MODEL

declare dllimport arm_aapcs_vfpcc void @initialise(ptr)

define dllexport arm_aapcs_vfpcc signext i8 @function(i32 %offset) #0 {
entry:
  %buffer = alloca [4096 x i8], align 1
  call arm_aapcs_vfpcc void @initialise(ptr %buffer)
  %arrayidx = getelementptr inbounds [4096 x i8], ptr %buffer, i32 0, i32 %offset
  %0 = load i8, ptr %arrayidx, align 1
  ret i8 %0
}

attributes #0 = { "stack-probe-size"="8096" }

; CHECK-DEFAULT-CODE-MODEL-NOT: __chkstk
; CHECK-DEFAULT-CODE-MODEL: sub.w sp, sp, #4096

; CHECK-LARGE-CODE-MODEL-NOT: movw r12, :lower16:__chkstk
; CHECK-LARGE-CODE-MODEL-NOT: movt r12, :upper16:__chkstk
; CHECK-LARGE-CODE-MODEL: sub.w sp, sp, #4096

