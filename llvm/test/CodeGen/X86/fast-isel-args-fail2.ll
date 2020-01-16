; RUN: not llc < %s -fast-isel -fast-isel-abort=2 -mtriple=x86_64-apple-darwin10
; REQUIRES: asserts

%struct.s0 = type { x86_fp80, x86_fp80 }

; FastISel cannot handle this case yet. Make sure that we abort.
define i8* @args_fail(%struct.s0* byval nocapture readonly align 16 %y) {
  %1 = bitcast %struct.s0* %y to i8*
  ret i8* %1
}
