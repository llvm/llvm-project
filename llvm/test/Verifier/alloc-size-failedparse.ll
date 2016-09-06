; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; We handle allocsize with identical args in the parser, rather than the
; verifier. So, a seperate test is needed.

; CHECK: 'allocsize' indices can't refer to the same parameter
declare i8* @a(i32, i32) allocsize(0, 0)
