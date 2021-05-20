; Check upgrade is removing the incompatible attributes on function types.

; RUN: llvm-dis < %s.bc | FileCheck %s

; CHECK: define i8 @f(i8 %0, i8 %1)
define align 8 i8 @f(i8 align 8 %0, i8 align 8 %1) {
  ret i8 0
}

; CHECK: declare i8 @f2(i8, i8, ...)
declare align 8 i8 @f2(i8 align 8, i8 align 8, ...)

declare i32* @"personality_function"()

define void @g() personality i32* ()* @"personality_function" {
; CHECK: call i8 @f(i8 0, i8 1)
  %1 = call align 8 i8 @f(i8 align 8 0, i8 align 8 1);
; CHECK: call i8 (i8, i8, ...) @f2(i8 0, i8 1, i8 2)
  %2 = call align 8 i8(i8, i8, ...) @f2(i8 align 8 0, i8 align 8 1, i8 align 8 2);
; CHECK: invoke i8 @f(i8 0, i8 1)
  %3 = invoke align 8 i8 @f(i8 align 8 0, i8 align 8 1) to label %cont unwind label %cleanup

cont:
  ret void

cleanup:
  %4 = landingpad i8 cleanup
  ret void
}
