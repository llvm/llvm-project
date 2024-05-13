; RUN: not --crash llc < %s -fast-isel -fast-isel-abort=2 -mtriple=x86_64-apple-darwin10

%struct.s0 = type { x86_fp80, x86_fp80 }

; FastISel cannot handle this case yet. Make sure that we abort.
define ptr @args_fail(ptr byval(%struct.s0) nocapture readonly align 16 %y) {
  ret ptr %y
}
