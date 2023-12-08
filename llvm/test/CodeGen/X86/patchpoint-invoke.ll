; RUN: llc -mtriple=x86_64-unknown-linux -mcpu=corei7                             < %s | FileCheck %s

; Test invoking of patchpoints
;
define i64 @patchpoint_invoke(i64 %p1, i64 %p2) personality ptr @__gxx_personality_v0 {
entry:
; CHECK-LABEL: patchpoint_invoke:
; CHECK-NEXT:  [[FUNC_BEGIN:.L.*]]:
; CHECK-NEXT: .cfi_startproc
; CHECK:      .cfi_lsda 3, [[EXCEPTION_LABEL:.L[^ ]*]]
; CHECK:      pushq %rbp

; Unfortunately, hardcode the name of the label that begins the patchpoint:
; CHECK:      .Ltmp0:
; CHECK:      movabsq $-559038736, %r11
; CHECK-NEXT: callq *%r11
; CHECK-NEXT: xchgw %ax, %ax
; CHECK-NEXT: [[PP_END:.L.*]]:
; CHECK:      ret
  %resolveCall = inttoptr i64 -559038736 to ptr
  %result = invoke i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 15, ptr %resolveCall, i32 1, i64 %p1, i64 %p2)
            to label %success unwind label %threw

success:
  ret i64 %result

threw:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i64 0
}

; Verify that the exception table was emitted:
; CHECK:      [[EXCEPTION_LABEL]]:
; CHECK-NEXT: .byte 255
; CHECK-NEXT: .byte 3
; CHECK-NEXT: .uleb128 .Lttbase{{[0-9]+}}-[[TTBASEREF:.Lttbaseref[0-9]+]]
; CHECK-NEXT: [[TTBASEREF]]:
; CHECK-NEXT: .byte 1
; CHECK-NEXT: .uleb128 .Lcst_end{{[0-9]+}}-[[CST_BEGIN:.Lcst_begin[0-9]+]]
; CHECK-NEXT: [[CST_BEGIN]]:
; Verify that the unwind data covers the entire patchpoint region:
; CHECK-NEXT: .uleb128 .Ltmp0-[[FUNC_BEGIN]]
; CHECK-NEXT: .uleb128 [[PP_END]]-.Ltmp0


; Verify that the stackmap section got emitted:
; CHECK-LABEL: __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 1
; Num LargeConstants
; CHECK-NEXT:   .long 0
; Num Callsites
; CHECK-NEXT:   .long 1
; CHECK-NEXT:   .quad patchpoint_invoke


declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)
declare i32 @__gxx_personality_v0(...)
