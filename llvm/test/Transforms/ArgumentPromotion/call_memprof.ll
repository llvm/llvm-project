; RUN: opt -passes=argpromotion -S < %s | FileCheck %s

; Checks if !memprof and !callsite metadata are preserved in argpromotion.

define internal i32 @test(ptr %p) {
  %v = load i32, ptr %p
  ret i32 %v
}

define void @caller(ptr %p) {
; CHECK: call i32 @test(i32 %{{.*}}), !memprof ![[MEMPROF:[0-9]+]], !callsite ![[CALLSITE:[0-9]+]]
  call i32 @test(ptr %p), !memprof !0, !callsite !2
  ret void
}

; CHECK: ![[MEMPROF]] = !{![[MIB:[0-9]+]]}
; CHECK: ![[MIB]] = !{![[CALLSITE]], !"cold"}
; CHECK: ![[CALLSITE]] = !{i64 123}
!0 = !{!1}
!1 = !{!2, !"cold"}
!2 = !{i64 123}
