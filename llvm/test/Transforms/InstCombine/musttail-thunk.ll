; RUN: opt -instcombine -S < %s | FileCheck %s

; These are both direct calls, but make sure instcombine leaves the casts
; alone.

define i32 @call_thunk(i32 %x, i32 %y) {
  %r = call i32 bitcast (void (i32, ...)* @inc_first_arg_thunk to i32 (i32, i32)*)(i32 %x, i32 %y)
  ret i32 %r
}

; CHECK-LABEL: define i32 @call_thunk(i32 %x, i32 %y)
; CHECK:   %r = call i32 bitcast (void (i32, ...)* @inc_first_arg_thunk to i32 (i32, i32)*)(i32 %x, i32 %y)
; CHECK:   ret i32 %r

define internal void @inc_first_arg_thunk(i32 %arg1, ...) #0 {
entry:
  %inc = add i32 %arg1, 1
  musttail call void (i32, ...) bitcast (i32 (i32, i32)* @plus to void (i32, ...)*)(i32 %inc, ...)
  ret void
}

; CHECK-LABEL: define internal void @inc_first_arg_thunk(i32 %arg1, ...) #0
; CHECK:   %inc = add i32 %arg1, 1
; CHECK:   musttail call void (i32, ...) bitcast (i32 (i32, i32)* @plus to void (i32, ...)*)(i32 %inc, ...)
; CHECK:   ret void

define internal i32 @plus(i32 %x, i32 %y) {
  %r = add i32 %x, %y
  ret i32 %r
}

attributes #0 = { "thunk" }
