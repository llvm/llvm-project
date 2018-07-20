; RUN: llc -mtriple=s390x-linux-gnu < %s | FileCheck %s

; Test invoking of patchpoints
;
define i64 @patchpoint_invoke(i64 %p1, i64 %p2) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK-LABEL: patchpoint_invoke:
; CHECK-NEXT:  [[FUNC_BEGIN:.L.*]]:
; CHECK-NEXT: .cfi_startproc
; CHECK:      .cfi_lsda 0, [[EXCEPTION_LABEL:.L[^ ]*]]
; CHECK:      aghi %r15, -160

; Unfortunately, hardcode the name of the label that begins the patchpoint:
; CHECK:      .Ltmp0:
; CHECK:      llilf   %r1, 559038736
; CHECK-NEXT: basr    %r14, %r1
; CHECK-NEXT: bcr     0, %r0
; CHECK-NEXT: [[PP_END:.L.*]]:
; CHECK:      br %r14
  %resolveCall = inttoptr i64 559038736 to i8*
  %result = invoke i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 2, i32 10, i8* %resolveCall, i32 1, i64 %p1, i64 %p2)
            to label %success unwind label %threw

success:
  ret i64 %result

threw:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  ret i64 0
}

; Verify that the exception table was emitted:
; CHECK:      [[EXCEPTION_LABEL]]:
; CHECK-NEXT: .byte 255
; CHECK-NEXT: .byte 0
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
declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
declare i32 @__gxx_personality_v0(...)
