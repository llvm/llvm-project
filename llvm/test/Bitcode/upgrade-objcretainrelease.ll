; Test that comment token for objc retain release is upgraded from '#' to ';'
;
; RUN: llvm-dis < %s.bc | FileCheck %s

; CHECK: "mov\09fp, fp\09\09; marker for objc_retainAutoreleaseReturnValue"

