; FIXME: Add tests for global-isel/fast-isel.

; RUN: llc < %s -mtriple=arm64-windows | FileCheck %s

; Returns <= 8 bytes should be in X0.
%struct.S1 = type { i32, i32 }
define dso_local i64 @"?f1"() {
entry:
; CHECK-LABEL: f1
; CHECK-DAG: str xzr, [sp, #8]
; CHECK-DAG: mov x0, xzr

  %retval = alloca %struct.S1, align 4
  store i32 0, ptr %retval, align 4
  %b = getelementptr inbounds %struct.S1, ptr %retval, i32 0, i32 1
  store i32 0, ptr %b, align 4
  %0 = load i64, ptr %retval, align 4
  ret i64 %0
}

; Returns <= 16 bytes should be in X0/X1.
%struct.S2 = type { i32, i32, i32, i32 }
define dso_local [2 x i64] @"?f2"() {
entry:
; FIXME: Missed optimization, the entire SP push/pop could be removed
; CHECK-LABEL: f2
; CHECK:         sub     sp, sp, #16
; CHECK-NEXT:    .seh_stackalloc 16
; CHECK-NEXT:    .seh_endprologue
; CHECK-DAG:     stp     xzr, xzr, [sp]
; CHECK-DAG:     mov     x0, xzr
; CHECK-DAG:     mov     x1, xzr
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    add     sp, sp, #16

  %retval = alloca %struct.S2, align 4
  store i32 0, ptr %retval, align 4
  %b = getelementptr inbounds %struct.S2, ptr %retval, i32 0, i32 1
  store i32 0, ptr %b, align 4
  %c = getelementptr inbounds %struct.S2, ptr %retval, i32 0, i32 2
  store i32 0, ptr %c, align 4
  %d = getelementptr inbounds %struct.S2, ptr %retval, i32 0, i32 3
  store i32 0, ptr %d, align 4
  %0 = load [2 x i64], ptr %retval, align 4
  ret [2 x i64] %0
}

; Arguments > 16 bytes should be passed in X8.
%struct.S3 = type { i32, i32, i32, i32, i32 }
define dso_local void @"?f3"(ptr noalias sret(%struct.S3) %agg.result) {
entry:
; CHECK-LABEL: f3
; CHECK: stp xzr, xzr, [x8]
; CHECK: str wzr, [x8, #16]

  store i32 0, ptr %agg.result, align 4
  %b = getelementptr inbounds %struct.S3, ptr %agg.result, i32 0, i32 1
  store i32 0, ptr %b, align 4
  %c = getelementptr inbounds %struct.S3, ptr %agg.result, i32 0, i32 2
  store i32 0, ptr %c, align 4
  %d = getelementptr inbounds %struct.S3, ptr %agg.result, i32 0, i32 3
  store i32 0, ptr %d, align 4
  %e = getelementptr inbounds %struct.S3, ptr %agg.result, i32 0, i32 4
  store i32 0, ptr %e, align 4
  ret void
}

; InReg arguments to non-instance methods must be passed in X0 and returns in
; X0.
%class.B = type { i32 }
define dso_local void @"?f4"(ptr inreg noalias nocapture sret(%class.B) %agg.result) {
entry:
; CHECK-LABEL: f4
; CHECK: mov w8, #1
; CHECK: str w8, [x0]
  store i32 1, ptr %agg.result, align 4
  ret void
}

; InReg arguments to instance methods must be passed in X1 and returns in X0.
%class.C = type { i8 }
%class.A = type { i8 }

define dso_local void @"?inst@C"(ptr %this, ptr inreg noalias sret(%class.A) %agg.result) {
entry:
; CHECK-LABEL: inst@C
; CHECK-DAG: mov x0, x1
; CHECK-DAG: str x8, [sp, #8]

  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; The following tests correspond to tests in
; clang/test/CodeGenCXX/microsoft-abi-sret-and-byval.cpp

; Pod is a trivial HFA
%struct.Pod = type { [2 x double] }
; Not an aggregate according to C++14 spec => not HFA according to MSVC
%struct.NotCXX14Aggregate  = type { %struct.Pod }
; NotPod is a C++14 aggregate. But not HFA, because it contains
; NotCXX14Aggregate (which itself is not HFA because it's not a C++14
; aggregate).
%struct.NotPod = type { %struct.NotCXX14Aggregate }

; CHECK-LABEL: copy_pod:
define dso_local %struct.Pod @copy_pod(ptr %x) {
  %x1 = load %struct.Pod, ptr %x, align 8
  ret %struct.Pod %x1
  ; CHECK: ldp d0, d1, [x0]
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

; CHECK-LABEL: copy_notcxx14aggregate:
define dso_local void
@copy_notcxx14aggregate(ptr inreg noalias sret(%struct.NotCXX14Aggregate) align 8 %agg.result,
                        ptr %x) {
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.result, ptr align 8 %x, i64 16, i1 false)
  ret void
  ; CHECK: str q0, [x0]
}

; CHECK-LABEL: copy_notpod:
define dso_local [2 x i64] @copy_notpod(ptr %x) {
  %x2 = load [2 x i64], ptr %x
  ret [2 x i64] %x2
  ; CHECK: ldp x8, x1, [x0]
  ; CHECK: mov x0, x8
}

@Pod = external global %struct.Pod

; CHECK-LABEL: call_copy_pod:
define void @call_copy_pod() {
  %x = call %struct.Pod @copy_pod(ptr @Pod)
  store %struct.Pod %x, ptr @Pod
  ret void
  ; CHECK: bl copy_pod
  ; CHECK-NEXT: str d0, [{{.*}}]
  ; CHECK-NEXT: str d1, [{{.*}}]
}

@NotCXX14Aggregate = external global %struct.NotCXX14Aggregate

; CHECK-LABEL: call_copy_notcxx14aggregate:
define void @call_copy_notcxx14aggregate() {
  %x = alloca %struct.NotCXX14Aggregate
  call void @copy_notcxx14aggregate(ptr %x, ptr @NotCXX14Aggregate)
  %x1 = load %struct.NotCXX14Aggregate, ptr %x
  store %struct.NotCXX14Aggregate %x1, ptr @NotCXX14Aggregate
  ret void
  ; CHECK: bl copy_notcxx14aggregate
  ; CHECK-NEXT: ldp {{.*}}, {{.*}}, [sp]
}

@NotPod = external global %struct.NotPod

; CHECK-LABEL: call_copy_notpod:
define void @call_copy_notpod() {
  %x = call [2 x i64] @copy_notpod(ptr @NotPod)
  store [2 x i64] %x, ptr @NotPod
  ret void
  ; CHECK: bl copy_notpod
  ; CHECK-NEXT: stp x0, x1, [{{.*}}]
}

; We shouldn't return the argument
; when it has only inreg attribute
define i64 @foobar(ptr inreg %0) {
; CHECK-LABEL: foobar:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ldr x0, [x0]
; CHECK-NEXT:    ret
entry:
  %1 = load i64, ptr %0
  ret i64 %1
}
