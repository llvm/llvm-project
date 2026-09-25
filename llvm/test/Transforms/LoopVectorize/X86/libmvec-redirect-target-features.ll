; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @generic(ptr noalias %out, ptr noalias %in) #0 {
; CHECK-LABEL: define void @generic(
; CHECK-NOT: call {{.*}}@custom_avx2
; CHECK: call double @foo
; CHECK-NOT: call {{.*}}@custom_avx2
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @foo(double %d) #9
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
; CHECK-NOT: call {{.*}}@custom_avx2
; CHECK: call double @foo
; CHECK-NOT: call {{.*}}@custom_avx2
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @foo(double %d) #9
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
; CHECK: call <4 x double> @custom_avx2
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @foo(double %d) #9
  %f = fptrunc double %r to float
  %q = getelementptr float, ptr %out, i64 %i
  store float %f, ptr %q, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @overwide_sse(ptr noalias %out, ptr noalias %in) #2 {
; CHECK-LABEL: define void @overwide_sse(
; CHECK-NOT: call {{.*}}@custom_sse
; CHECK: call double @foo
; CHECK-NOT: call {{.*}}@custom_sse
; CHECK: ret void
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %in, i64 %i
  %x = load float, ptr %p, align 4
  %d = fpext float %x to double
  %r = call double @foo(double %d) #10
  %f = fptrunc double %r to float
  %q = getelementptr float, ptr %out, i64 %i
  store float %f, ptr %q, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare double @foo(double) nounwind memory(none)
declare <4 x double> @custom_avx2(<4 x double>)
attributes #0 = { "target-cpu"="x86-64" }
attributes #1 = { "target-cpu"="sandybridge" }
attributes #9 = { "vector-function-abi-variant"="_ZGVdN4v_foo(custom_avx2)" }

attributes #2 = { "target-cpu"="haswell" }

declare <4 x double> @custom_sse(<4 x double>)
attributes #10 = { "vector-function-abi-variant"="_ZGVbN4v_foo(custom_sse)" }
