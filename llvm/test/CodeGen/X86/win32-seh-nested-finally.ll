; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

define void @nested_finally() #0 personality i8* bitcast (i32 (...)* @_except_handler3 to i8*) {
entry:
  invoke void @f(i32 1) #3
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void @f(i32 2) #3
          to label %invoke.cont.1 unwind label %ehcleanup.3

invoke.cont.1:                                    ; preds = %invoke.cont
  call void @f(i32 3) #3
  ret void

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  invoke void @f(i32 2) #3
          to label %invoke.cont.2 unwind label %ehcleanup.3

invoke.cont.2:                                    ; preds = %ehcleanup
  cleanupret from %0 unwind label %ehcleanup.3

ehcleanup.3:                                      ; preds = %invoke.cont.2, %ehcleanup.end, %invoke.cont
  %1 = cleanuppad within none []
  call void @f(i32 3) #3
  cleanupret from %1 unwind to caller
}

declare void @f(i32) #0

declare i32 @_except_handler3(...)

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { noinline }

; CHECK: _nested_finally:
; CHECK: movl $-1, -[[state:[0-9]+]](%ebp)
; CHECK: movl {{.*}}, %fs:0
; CHECK: movl $1, -[[state]](%ebp)
; CHECK: movl $1, (%esp)
; CHECK: calll _f
; CHECK: movl $0, -[[state]](%ebp)
; CHECK: movl $2, (%esp)
; CHECK: calll _f
; CHECK: movl $-1, -[[state]](%ebp)
; CHECK: movl $3, (%esp)
; CHECK: calll _f
; CHECK: retl

; CHECK: LBB0_[[inner:[0-9]+]]: # %ehcleanup
; CHECK: pushl %ebp
; CHECK: addl $12, %ebp
; CHECK: movl $0, -[[state]](%ebp)
; CHECK: movl $2, (%esp)
; CHECK: calll _f
; CHECK: popl %ebp
; CHECK: retl

; CHECK: LBB0_[[outer:[0-9]+]]: # %ehcleanup.3
; CHECK: pushl %ebp
; CHECK: addl $12, %ebp
; CHECK: movl $-1, -[[state]](%ebp)
; CHECK: movl $3, (%esp)
; CHECK: calll _f
; CHECK: popl %ebp
; CHECK: retl

; CHECK: L__ehtable$nested_finally:
; CHECK:        .long   -1
; CHECK:        .long   0
; CHECK:        .long   LBB0_[[outer]]
; CHECK:        .long   0
; CHECK:        .long   0
; CHECK:        .long   LBB0_[[inner]]
