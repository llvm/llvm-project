; Test that comment token for objc retain release is upgraded from '#' to ';'
;
; RUN: llvm-dis < %s.bc | FileCheck %s

; CHECK: !llvm.module.flags = !{!0}
; CHECK: !0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov\09fp, fp\09\09; marker for objc_retainAutoreleaseReturnValue"}
