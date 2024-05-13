; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Check the error message on using ", ..." when we can't actually forward
; varargs.

%struct.A = type { i32 }

declare ptr @f(ptr, ...)

define ptr @f_thunk(ptr %this) {
  %rv = musttail call ptr (ptr, ...) @f(ptr %this, ...)
; CHECK: error: unexpected ellipsis in argument list for musttail call in non-varargs function
  ret ptr %rv
}
