; REQUIRES: arm-registered-target
; RUN: opt < %s -passes=cg-profile -S | FileCheck %s

;; Calls to intrinsics are not edges of the call graph profile, even when TTI
;; says they are lowered to calls, as ARM does for these.

target datalayout = "e-m:w-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-pc-windows-msvc"

declare double @llvm.exp2.f64(double)
declare double @llvm.pow.f64(double, double)
declare double @llvm.powi.f64.i32(double, i32)
declare <4 x i32> @llvm.masked.load.v4i32.p0(ptr, <4 x i1>, <4 x i32>)
declare void @g()

define double @f(double %x, i32 %n, ptr %p, <4 x i1> %m, ptr %q) !prof !0 {
  call void @g()
  %a = call double @llvm.exp2.f64(double %x)
  %b = call double @llvm.pow.f64(double %a, double %x)
  %c = call double @llvm.powi.f64.i32(double %b, i32 %n)
  %v = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 %p, <4 x i1> %m, <4 x i32> zeroinitializer)
  store <4 x i32> %v, ptr %q
  ret double %c
}

!0 = !{!"function_entry_count", i64 100}

;; The profile has exactly one edge, the one to @g.
; CHECK: !llvm.module.flags = !{![[CGPROF:[0-9]+]]}
; CHECK: ![[CGPROF]] = !{i32 5, !"CG Profile", ![[EDGES:[0-9]+]]}
; CHECK: ![[EDGES]] = distinct !{![[E0:[0-9]+]]}
; CHECK: ![[E0]] = !{ptr @f, ptr @g, i64 100}
; CHECK-NOT: llvm.
