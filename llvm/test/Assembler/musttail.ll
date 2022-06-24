; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s

; Check that the ellipsis round trips.

%struct.A = type { i32 }

declare i8* @f(i8*, ...)

define i8* @f_thunk(i8* %this, ...) {
  %rv = musttail call i8* (i8*, ...) @f(i8* %this, ...)
  ret i8* %rv
}
; CHECK-LABEL: define i8* @f_thunk(i8* %this, ...)
; CHECK: %rv = musttail call i8* (i8*, ...) @f(i8* %this, ...)
