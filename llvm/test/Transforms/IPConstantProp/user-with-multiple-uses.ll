; RUN: opt < %s -S -ipsccp | FileCheck %s
; PR5596

; IPSCCP should propagate the 0 argument, eliminate the switch, and propagate
; the result.

; CHECK: define i32 @main() #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT: %call2 = tail call i32 @wwrite(i64 0) [[NUW:#[0-9]+]]
; CHECK-NEXT: ret i32 123

define i32 @main() noreturn nounwind {
entry:
  %call2 = tail call i32 @wwrite(i64 0) nounwind
  ret i32 %call2
}

define internal i32 @wwrite(i64 %i) nounwind readnone {
entry:
  switch i64 %i, label %sw.default [
    i64 3, label %return
    i64 10, label %return
  ]

sw.default:
  ret i32 123

return:
  ret i32 0
}

; CHECK: attributes #0 = { noreturn nounwind }
; CHECK: attributes #1 = { nounwind readnone }
; CHECK: attributes [[NUW]] = { nounwind }
