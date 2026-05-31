; AIR GPU JIT rejects i64 air.simd_shuffle; rewrite as bitcast-to-v2i32 +
; vector shuffle. Empirically the JIT accepts the v2i32 vector form (see
; PASS_GUARDS.md G3.6).
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK: bitcast i64 %v to <2 x i32>
; CHECK: call <2 x i32> @air.simd_shuffle.v2i32(<2 x i32>
; CHECK: bitcast <2 x i32> {{.*}} to i64
; CHECK-NOT: call i64 @air.simd_shuffle.i64

declare i64 @air.simd_shuffle.i64(i64, i16)

define void @kernel(i64 %v, i16 %off, ptr %out) {
entry:
  %s = call i64 @air.simd_shuffle.i64(i64 %v, i16 %off)
  store i64 %s, ptr %out
  ret void
}
