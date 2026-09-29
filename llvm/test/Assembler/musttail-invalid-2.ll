; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Check the error message on skipping ", ..." at the end of a musttail call argument list.

%struct.A = type { i32 }

declare ptr @f(ptr, ...)

define ptr @f_thunk(ptr %this, ...) {
  %rv = musttail call ptr (ptr, ...) @f(ptr %this)
; CHECK: error: expected '...' at end of argument list for musttail call in varargs function
  ret ptr %rv
}
