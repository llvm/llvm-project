; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal -attributor-annotate-decl-cs -S < %s | FileCheck %s

; Checks if !memprof and !callsite metadata are preserved in attributor.

declare void @external_sink()

define void @caller() {
; CHECK: call void @test(), !memprof ![[MEMPROF:[0-9]+]], !callsite ![[CALLSITE:[0-9]+]]
  call void @test(i32 1), !memprof !0, !callsite !2
  ret void
}

define internal void @test(i32 %dead_arg) {
  call void @external_sink()
  ret void
}

; CHECK: ![[MEMPROF]] = !{![[MIB:[0-9]+]]}
; CHECK: ![[MIB]] = !{![[CALLSITE]], !"cold"}
; CHECK: ![[CALLSITE]] = !{i64 123}
!0 = !{!1}
!1 = !{!2, !"cold"}
!2 = !{i64 123}
