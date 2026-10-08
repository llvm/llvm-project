; RUN: opt -passes=inject-tli-mappings,slp-vectorizer -vector-library=LIBMVEC -slp-threshold=-1000 -S %s | FileCheck %s
; An AVX register can hold the arguments, but the d ABI variant requires AVX2.
target triple = "x86_64-unknown-linux-gnu"

define void @avx(ptr noalias %out, ptr noalias %in) #1 {
; CHECK-LABEL: define void @avx(
; CHECK-NOT: call {{.*}}@_ZGVd
; CHECK: ret void
  %p0 = getelementptr double, ptr %in, i64 0
  %x0 = load double, ptr %p0, align 8
  %r0 = call fast double @erf(double %x0)
  %p1 = getelementptr double, ptr %in, i64 1
  %x1 = load double, ptr %p1, align 8
  %r1 = call fast double @erf(double %x1)
  %p2 = getelementptr double, ptr %in, i64 2
  %x2 = load double, ptr %p2, align 8
  %r2 = call fast double @erf(double %x2)
  %p3 = getelementptr double, ptr %in, i64 3
  %x3 = load double, ptr %p3, align 8
  %r3 = call fast double @erf(double %x3)
  %q0 = getelementptr double, ptr %out, i64 0
  store double %r0, ptr %q0, align 8
  %q1 = getelementptr double, ptr %out, i64 1
  store double %r1, ptr %q1, align 8
  %q2 = getelementptr double, ptr %out, i64 2
  store double %r2, ptr %q2, align 8
  %q3 = getelementptr double, ptr %out, i64 3
  store double %r3, ptr %q3, align 8
  ret void
}

define void @avx2(ptr noalias %out, ptr noalias %in) #2 {
; CHECK-LABEL: define void @avx2(
; CHECK: call fast <4 x double> @_ZGVdN4v_erf
; CHECK: ret void
  %p0 = getelementptr double, ptr %in, i64 0
  %x0 = load double, ptr %p0, align 8
  %r0 = call fast double @erf(double %x0)
  %p1 = getelementptr double, ptr %in, i64 1
  %x1 = load double, ptr %p1, align 8
  %r1 = call fast double @erf(double %x1)
  %p2 = getelementptr double, ptr %in, i64 2
  %x2 = load double, ptr %p2, align 8
  %r2 = call fast double @erf(double %x2)
  %p3 = getelementptr double, ptr %in, i64 3
  %x3 = load double, ptr %p3, align 8
  %r3 = call fast double @erf(double %x3)
  %q0 = getelementptr double, ptr %out, i64 0
  store double %r0, ptr %q0, align 8
  %q1 = getelementptr double, ptr %out, i64 1
  store double %r1, ptr %q1, align 8
  %q2 = getelementptr double, ptr %out, i64 2
  store double %r2, ptr %q2, align 8
  %q3 = getelementptr double, ptr %out, i64 3
  store double %r3, ptr %q3, align 8
  ret void
}

declare double @erf(double) nounwind willreturn memory(none)
attributes #1 = { "target-cpu"="sandybridge" }
attributes #2 = { "target-cpu"="haswell" }
