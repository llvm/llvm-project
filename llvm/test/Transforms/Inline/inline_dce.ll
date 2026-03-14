; This checks to ensure that the inline pass deletes functions if they get
; inlined into all of their callers.

; RUN: opt < %s -passes=inline -S | \
; RUN:   not grep @reallysmall

define internal i32 @reallysmall(i32 %A) {
; CHECK-NOT: @reallysmall
entry:
  ret i32 %A
}

define void @caller1() {
; CHECK-LABEL: define void @caller1()
entry:
  call i32 @reallysmall(i32 5)
; CHECK-NOT: call
  ret void
}

define void @caller2(i32 %A) {
; CHECK-LABEL: define void @caller2(i32 %A)
entry:
  call i32 @reallysmall(i32 %A)
; CHECK-NOT: call
  ret void
}

define i32 @caller3(i32 %A) {
; CHECK-LABEL: define void @caller3(i32 %A)
entry:
  %B = call i32 @reallysmall(i32 %A)
; CHECK-NOT: call
  ret i32 %B
}

