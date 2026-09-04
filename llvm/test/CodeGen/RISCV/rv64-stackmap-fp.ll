; RUN: llc -mtriple=riscv64 -mattr=+d,+zfh < %s | FileCheck %s

; CHECK-LABEL:  .section	.llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte   3
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   0
; Num Functions
; CHECK-NEXT:   .word   1
; Num LargeConstants
; CHECK-NEXT:   .word   0
; Num Callsites
; CHECK-NEXT:   .word   1

; Functions and stack size
; CHECK-NEXT:   .quad   liveArgs
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   1

; Spilled stack map values.
;
; Verify 3 stack map entries.
;
; CHECK-LABEL:  .word   .L{{.*}}-liveArgs
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   25
;
; Check that at least one is a spilled entry from SP.
; Location: Indirect SP + ...
; CHECK:        .byte   3
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word
define void @liveArgs(double %arg0, double %arg1, double %arg2, double %arg3, double %arg4, double %arg5, double %arg6, double %arg7, double %arg8, double %arg9, double %arg10, double %arg11, double %arg12, double %arg13, double %arg14, double %arg15, double %arg16, double %arg17, double %arg18, double %arg19, double %arg20, double %arg21, double %arg22, double %arg23, half %arg24, half %arg25, half %arg26, half %arg27, half %arg28, bfloat %arg29) {
entry:
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 11, i32 28, ptr null, i32 5, double %arg0, double %arg1, double %arg2, double %arg3, double %arg4, double %arg5, double %arg6, double %arg7, double %arg8, double %arg9, double %arg10, double %arg11, double %arg12, double %arg13, double %arg14, double %arg15, double %arg16, double %arg17, double %arg18, double %arg19, double %arg20, double %arg21, double %arg22, double %arg23, half %arg24, half %arg25, half %arg26, half %arg27, half %arg28, bfloat %arg29)
  ret void
}
