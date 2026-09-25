; RUN: opt -passes=inject-tli-mappings,loop-vectorize -vector-library=LIBMVEC -force-vector-width=4 -force-vector-interleave=1 -S %s | FileCheck %s
; The loop's elements are floats, but each library call operates on doubles.
target triple = "x86_64-unknown-linux-gnu"

define void @generic(ptr noalias %out, ptr noalias %in) #0 {
; CHECK-LABEL: define void @generic(
; CHECK-NOT: call {{.*}}@_ZGVd
; CHECK: call double @erf
; CHECK-NOT: call {{.*}}@_ZGVd
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @erf(double %d)
  %f = fptrunc double %r to float
  %q = getelementptr float, ptr %out, i64 %i
  store float %f, ptr %q, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @avx(ptr noalias %out, ptr noalias %in) #1 {
; CHECK-LABEL: define void @avx(
; CHECK-NOT: call {{.*}}@_ZGVd
; CHECK: call double @erf
; CHECK-NOT: call {{.*}}@_ZGVd
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @erf(double %d)
  %f = fptrunc double %r to float
  %q = getelementptr float, ptr %out, i64 %i
  store float %f, ptr %q, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @avx2(ptr noalias %out, ptr noalias %in) #2 {
; CHECK-LABEL: define void @avx2(
; CHECK: call <4 x double> @_ZGVdN4v_erf
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @erf(double %d)
  %f = fptrunc double %r to float
  %q = getelementptr float, ptr %out, i64 %i
  store float %f, ptr %q, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare double @erf(double) nounwind memory(none)
attributes #0 = { "target-cpu"="x86-64" }
attributes #1 = { "target-cpu"="sandybridge" }
attributes #2 = { "target-cpu"="haswell" }
attributes #3 = { "target-cpu"="skylake-avx512" "prefer-vector-width"="256" }
attributes #4 = { "target-cpu"="skylake-avx512" "prefer-vector-width"="512" }
attributes #5 = { "target-cpu"="haswell" "target-features"="-avx2" }
attributes #6 = { "target-cpu"="haswell" "prefer-vector-width"="128" }
