; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Check that intrinsic calls with mangling are error checked.
define void @foo(float %a, i32 %b) {
    ; CHECK: <stdin>:[[@LINE+1]]:18: error: invalid intrinsic name, expected @llvm.is.constant.f32
    %c = call i1 @llvm.is.constant.badsuffix(float %a)
    ret void
}
