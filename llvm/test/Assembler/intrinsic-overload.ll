; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Check that intrinsic calls without any mangling are converted to correct
; mangled forms. And mangled forms with correct mangling parse correctly.
define void @foo(float %a, i32 %b) {
    ; CHECK:  call i1 @llvm.is.constant.f32
    %c = call i1 @llvm.is.constant(float %a)

    ; CHECK: call i1 @llvm.is.constant.i32
    %d = call i1 @llvm.is.constant(i32 %b)

    ; CHECK: call i1 @llvm.is.constant.i1
    %e = call i1 @llvm.is.constant.i1(i1 false)

    ret void
}
