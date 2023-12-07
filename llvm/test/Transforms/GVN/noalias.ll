; RUN: opt -passes=gvn -S < %s | FileCheck %s

define i32 @test1(ptr %p, ptr %q) {
; CHECK-LABEL: @test1(ptr %p, ptr %q)
; CHECK: load i32, ptr %p
; CHECK-NOT: noalias
; CHECK: %c = add i32 %a, %a
  %a = load i32, ptr %p, !noalias !3
  %b = load i32, ptr %p
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test2(ptr %p, ptr %q) {
; CHECK-LABEL: @test2(ptr %p, ptr %q)
; CHECK: load i32, ptr %p, align 4, !alias.scope ![[SCOPE1:[0-9]+]]
; CHECK: %c = add i32 %a, %a
  %a = load i32, ptr %p, !alias.scope !3
  %b = load i32, ptr %p, !alias.scope !3
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test3(ptr %p, ptr %q) {
; CHECK-LABEL: @test3(ptr %p, ptr %q)
; CHECK: load i32, ptr %p, align 4, !alias.scope ![[SCOPE2:[0-9]+]]
; CHECK: %c = add i32 %a, %a
  %a = load i32, ptr %p, !alias.scope !4
  %b = load i32, ptr %p, !alias.scope !5
  %c = add i32 %a, %b
  ret i32 %c
}

; CHECK:   ![[SCOPE1]] = !{!{{[0-9]+}}}
; CHECK:   ![[SCOPE2]] = !{!{{[0-9]+}}, !{{[0-9]+}}}
declare i32 @foo(ptr) readonly

!0 = distinct !{!0, !2, !"callee0: %a"}
!1 = distinct !{!1, !2, !"callee0: %b"}
!2 = distinct !{!2, !"callee0"}

!3 = !{!0}
!4 = !{!1}
!5 = !{!0, !1}
