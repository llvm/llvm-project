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
  %a = getelementptr inbounds %struct.S1, %struct.S1* %retval, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %b = getelementptr inbounds %struct.S1, %struct.S1* %retval, i32 0, i32 1
  store i32 0, i32* %b, align 4
  %0 = bitcast %struct.S1* %retval to i64*
  %1 = load i64, i64* %0, align 4
  ret i64 %1
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
  %a = getelementptr inbounds %struct.S2, %struct.S2* %retval, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %b = getelementptr inbounds %struct.S2, %struct.S2* %retval, i32 0, i32 1
  store i32 0, i32* %b, align 4
  %c = getelementptr inbounds %struct.S2, %struct.S2* %retval, i32 0, i32 2
  store i32 0, i32* %c, align 4
  %d = getelementptr inbounds %struct.S2, %struct.S2* %retval, i32 0, i32 3
  store i32 0, i32* %d, align 4
  %0 = bitcast %struct.S2* %retval to [2 x i64]*
  %1 = load [2 x i64], [2 x i64]* %0, align 4
  ret [2 x i64] %1
}

; Arguments > 16 bytes should be passed in X8.
%struct.S3 = type { i32, i32, i32, i32, i32 }
define dso_local void @"?f3"(%struct.S3* noalias sret(%struct.S3) %agg.result) {
entry:
; CHECK-LABEL: f3
; CHECK: stp xzr, xzr, [x8]
; CHECK: str wzr, [x8, #16]

  %a = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %b = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 1
  store i32 0, i32* %b, align 4
  %c = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 2
  store i32 0, i32* %c, align 4
  %d = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 3
  store i32 0, i32* %d, align 4
  %e = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 4
  store i32 0, i32* %e, align 4
  ret void
}

; InReg arguments to non-instance methods must be passed in X0 and returns in
; X0.
%class.B = type { i32 }
define dso_local void @"?f4"(%class.B* inreg noalias nocapture sret(%class.B) %agg.result) {
entry:
; CHECK-LABEL: f4
; CHECK: mov w8, #1
; CHECK: str w8, [x0]
  %X.i = getelementptr inbounds %class.B, %class.B* %agg.result, i64 0, i32 0
  store i32 1, i32* %X.i, align 4
  ret void
}

; InReg arguments to instance methods must be passed in X1 and returns in X0.
%class.C = type { i8 }
%class.A = type { i8 }

define dso_local void @"?inst@C"(%class.C* %this, %class.A* inreg noalias sret(%class.A) %agg.result) {
entry:
; CHECK-LABEL: inst@C
; CHECK-DAG: mov x0, x1
; CHECK-DAG: str x8, [sp, #8]

  %this.addr = alloca %class.C*, align 8
  store %class.C* %this, %class.C** %this.addr, align 8
  %this1 = load %class.C*, %class.C** %this.addr, align 8
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
define dso_local %struct.Pod @copy_pod(%struct.Pod* %x) {
  %x1 = load %struct.Pod, %struct.Pod* %x, align 8
  ret %struct.Pod %x1
  ; CHECK: ldp d0, d1, [x0]
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

; CHECK-LABEL: copy_notcxx14aggregate:
define dso_local void
@copy_notcxx14aggregate(%struct.NotCXX14Aggregate* inreg noalias sret(%struct.NotCXX14Aggregate) align 8 %agg.result,
                        %struct.NotCXX14Aggregate* %x) {
  %1 = bitcast %struct.NotCXX14Aggregate* %agg.result to i8*
  %2 = bitcast %struct.NotCXX14Aggregate* %x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %1, i8* align 8 %2, i64 16, i1 false)
  ret void
  ; CHECK: str q0, [x0]
}

; CHECK-LABEL: copy_notpod:
define dso_local [2 x i64] @copy_notpod(%struct.NotPod* %x) {
  %x1 = bitcast %struct.NotPod* %x to [2 x i64]*
  %x2 = load [2 x i64], [2 x i64]* %x1
  ret [2 x i64] %x2
  ; CHECK: ldp x8, x1, [x0]
  ; CHECK: mov x0, x8
}

@Pod = external global %struct.Pod

; CHECK-LABEL: call_copy_pod:
define void @call_copy_pod() {
  %x = call %struct.Pod @copy_pod(%struct.Pod* @Pod)
  store %struct.Pod %x, %struct.Pod* @Pod
  ret void
  ; CHECK: bl copy_pod
  ; CHECK-NEXT: str d0, [{{.*}}]
  ; CHECK-NEXT: str d1, [{{.*}}]
}

@NotCXX14Aggregate = external global %struct.NotCXX14Aggregate

; CHECK-LABEL: call_copy_notcxx14aggregate:
define void @call_copy_notcxx14aggregate() {
  %x = alloca %struct.NotCXX14Aggregate
  call void @copy_notcxx14aggregate(%struct.NotCXX14Aggregate* %x, %struct.NotCXX14Aggregate* @NotCXX14Aggregate)
  %x1 = load %struct.NotCXX14Aggregate, %struct.NotCXX14Aggregate* %x
  store %struct.NotCXX14Aggregate %x1, %struct.NotCXX14Aggregate* @NotCXX14Aggregate
  ret void
  ; CHECK: bl copy_notcxx14aggregate
  ; CHECK-NEXT: ldp {{.*}}, {{.*}}, [sp]
}

@NotPod = external global %struct.NotPod

; CHECK-LABEL: call_copy_notpod:
define void @call_copy_notpod() {
  %x = call [2 x i64] @copy_notpod(%struct.NotPod* @NotPod)
  %notpod = bitcast %struct.NotPod* @NotPod to [2 x i64]*
  store [2 x i64] %x, [2 x i64]* %notpod
  ret void
  ; CHECK: bl copy_notpod
  ; CHECK-NEXT: stp x0, x1, [{{.*}}]
}

; We shouldn't return the argument
; when it has only inreg attribute
define i64 @foobar(i64* inreg %0) {
; CHECK-LABEL: foobar:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ldr x0, [x0]
; CHECK-NEXT:    ret
entry:
  %1 = load i64, i64* %0
  ret i64 %1
}
