; RUN: opt -passes="default<O3>" -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

@v = internal unnamed_addr global i32 0, align 4
@p = common global ptr null, align 8


; This test checks that a number of loads and stores are eliminated,
; that can only be eliminated based on GlobalsAA information. As such,
; it tests that GlobalsAA information is retained until the passes
; that perform this optimization, and it protects against accidentally
; dropping the GlobalsAA information earlier in the pipeline, which
; has happened a few times.

; GlobalsAA invalidation might happen later in the FunctionPassManager
; pipeline than the optimization eliminating unnecessary loads/stores.
; Since GlobalsAA is a module-level analysis, any FunctionPass
; invalidating the GlobalsAA information will affect FunctionPass
; pipelines that execute later. For example, assume a FunctionPass1 |
; FunctionPass2 pipeline and 2 functions to be processed: f1 and f2.
; Assume furthermore that FunctionPass1 uses GlobalsAA info to do an
; optimization, and FunctionPass2 invalidates GlobalsAA. Assume the
; function passes run in the following order: FunctionPass1(f1),
; FunctionPass2(f1), FunctionPass1(f2), FunctionPass2(f2). Then
; FunctionPass1 will not be able to optimize f2, since GlobalsAA will
; have been invalidated in FuntionPass2(f1).

; To try and also test this scenario, there is an empty function
; before and after the function we're checking so that one of them
; will be processed by the whole set of FunctionPasses before @f. That
; will ensure that if the invalidation happens, it happens before the
; actual optimizations on @f start.
define void @bar() {
entry:
  ret void
}

; Function Attrs: norecurse nounwind
define void @f(i32 %n) {
entry:
  %0 = load i32, ptr @v, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @v, align 4
  %1 = load ptr, ptr @p, align 8
  store i32 %n, ptr %1, align 4
  %2 = load i32, ptr @v, align 4
  %inc1 = add nsw i32 %2, 1
  store i32 %inc1, ptr @v, align 4
  ret void
}

; check variable v is loaded/stored only once after optimization,
; which should be prove that globalsAA survives until the optimization
; that can use it to optimize away the duplicate load/stores on
; variable v.
; CHECK:     load i32, ptr @v, align 4
; CHECK:     store i32 {{.*}}, ptr @v, align 4
; CHECK-NOT: load i32, ptr @v, align 4
; CHECK-NOT:     store i32 {{.*}}, ptr @v, align 4

; Same as @bar above, in case the functions are processed in reverse order.
define void @bar2() {
entry:
  ret void
}
