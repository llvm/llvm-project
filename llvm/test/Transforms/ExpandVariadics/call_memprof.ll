; RUN: opt -mtriple=wasm32-unknown-unknown -S --passes=expand-variadics --expand-variadics-override=lowering < %s | FileCheck %s

declare void @sink(...)

define void @caller() {
; CHECK: call void @sink(ptr %{{.*}}), !memprof ![[MEMPROF:[0-9]+]], !callsite ![[CALLSITE:[0-9]+]]
  call void (...) @sink(), !memprof !0, !callsite !2
  ret void
}

; CHECK: ![[MEMPROF]] = !{![[MIB:[0-9]+]]}
; CHECK: ![[MIB]] = !{![[CALLSITE]], !"cold"}
; CHECK: ![[CALLSITE]] = !{i64 123}
!0 = !{!1}
!1 = !{!2, !"cold"}
!2 = !{i64 123}
