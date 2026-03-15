; RUN: not llc < %s -fast-isel -fast-isel-abort=2 -mtriple=x86_64-apple-darwin10 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: FastISel didn't lower all arguments: ptr (ptr) (in function: args_fail)

%struct.s0 = type { x86_fp80, x86_fp80 }

; FastISel cannot handle this case yet. Make sure that we abort.
define ptr @args_fail(ptr byval(%struct.s0) nocapture readonly align 16 %y) {
  ret ptr %y
}
