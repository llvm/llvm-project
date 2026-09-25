; RUN: opt -passes=inject-tli-mappings,loop-vectorize,replace-with-veclib -vector-library=AMDLIBM -force-vector-width=8 -force-vector-interleave=1 -S %s | FileCheck %s
; Eight floats occupy 256 bits, but a double-precision call at VF8 needs 512.
; Check both the loop vectorizer and subsequent intrinsic replacement.
target triple = "x86_64-unknown-linux-gnu"

define void @avx2(ptr noalias %out, ptr noalias %in) #0 {
; CHECK-LABEL: define void @avx2(
; CHECK-NOT: call {{.*}}@amd_vrd8_log
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @llvm.log.f64(double %d)
  %f = fptrunc double %r to float
  %q = getelementptr float, ptr %out, i64 %i
  store float %f, ptr %q, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @prefer256(ptr noalias %out, ptr noalias %in) #1 {
; CHECK-LABEL: define void @prefer256(
; CHECK-NOT: call {{.*}}@amd_vrd8_log
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @llvm.log.f64(double %d)
  %f = fptrunc double %r to float
  %q = getelementptr float, ptr %out, i64 %i
  store float %f, ptr %q, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @avx512(ptr noalias %out, ptr noalias %in) #2 {
; CHECK-LABEL: define void @avx512(
; CHECK: call <8 x double> @amd_vrd8_log
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @llvm.log.f64(double %d)
  %f = fptrunc double %r to float
  %q = getelementptr float, ptr %out, i64 %i
  store float %f, ptr %q, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare double @llvm.log.f64(double)
attributes #0 = { "target-cpu"="haswell" }
attributes #1 = { "target-cpu"="skylake-avx512" "prefer-vector-width"="256" }
attributes #2 = { "target-cpu"="skylake-avx512" "prefer-vector-width"="512" }
