; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Check that intrinsic calls with mangling are error checked.
; Mix good and bad calls to demonstrate that line number tracking is required
; in the parser to report correct line number for the second (bad) call.
define void @foo(float %a, i32 %b) {
    %c = call i1 @llvm.is.constant.i32(i32 0)
    ; CHECK: <stdin>:[[@LINE+1]]:18: error: invalid intrinsic name, expected @llvm.is.constant.f32
    %d = call i1 @llvm.is.constant.i32(float %a)
    %e = call i1 @llvm.is.constant.i1(i1 false)
    ret void
}
