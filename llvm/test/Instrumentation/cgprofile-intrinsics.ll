; REQUIRES: arm-registered-target
; RUN: opt < %s -passes=cg-profile -S | FileCheck %s

;; A call to an intrinsic is not an edge of the call graph profile, even when
;; TTI says it is lowered to a call, as ARM does for llvm.exp2.

target triple = "thumbv7-pc-windows-msvc"

declare double @llvm.exp2.f64(double)
declare void @g()

define double @f(double %x) !prof !0 {
  call void @g()
  %r = call double @llvm.exp2.f64(double %x)
  ret double %r
}

!0 = !{!"function_entry_count", i64 100}

;; The profile has exactly one edge, the one to @g.
; CHECK: !llvm.module.flags = !{![[CGPROF:[0-9]+]]}
; CHECK: ![[CGPROF]] = !{i32 5, !"CG Profile", ![[EDGES:[0-9]+]]}
; CHECK: ![[EDGES]] = distinct !{![[E0:[0-9]+]]}
; CHECK: ![[E0]] = !{ptr @f, ptr @g, i64 100}
