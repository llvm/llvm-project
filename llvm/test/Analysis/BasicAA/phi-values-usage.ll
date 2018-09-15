; RUN: opt -debug-pass=Executions -phi-values -memcpyopt -instcombine -disable-output < %s 2>&1 | FileCheck %s

; Check that phi values is not run when it's not already available, and that
; basicaa is freed after a pass that preserves CFG.

; CHECK: Executing Pass 'Phi Values Analysis'
; CHECK: Executing Pass 'Basic Alias Analysis (stateless AA impl)'
; CHECK: Executing Pass 'Memory Dependence Analysis'
; CHECK: Executing Pass 'MemCpy Optimization'
; CHECK-DAG: Freeing Pass 'MemCpy Optimization'
; CHECK-DAG: Freeing Pass 'Phi Values Analysis'
; CHECK-DAG: Freeing Pass 'Memory Dependence Analysis'
; CHECK-DAG: Freeing Pass 'Basic Alias Analysis (stateless AA impl)'
; CHECK-NOT: Executing Pass 'Phi Values Analysis'
; CHECK: Executing Pass 'Basic Alias Analysis (stateless AA impl)'
; CHECK: Executing Pass 'Combine redundant instructions'

declare void @otherfn([4 x i8]*)
declare i32 @__gxx_personality_v0(...)

; This function is one where if we didn't free basicaa after memcpyopt then the
; usage of basicaa in instcombine would cause a segfault due to stale phi-values
; results being used.
define void @fn(i8* %this, i64* %ptr) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %arr = alloca [4 x i8], align 8
  %gep1 = getelementptr inbounds [4 x i8], [4 x i8]* %arr, i64 0, i32 0
  br i1 undef, label %then, label %if

if:
  br label %then

then:
  %phi = phi i64* [ %ptr, %if ], [ null, %entry ]
  store i8 1, i8* %gep1, align 8
  %load = load i64, i64* %phi, align 8
  %gep2 = getelementptr inbounds i8, i8* undef, i64 %load
  %gep3 = getelementptr inbounds i8, i8* %gep2, i64 40
  invoke i32 undef(i8* undef)
     to label %invoke unwind label %lpad

invoke:
  unreachable

lpad:
  landingpad { i8*, i32 }
     catch i8* null
  call void @otherfn([4 x i8]* nonnull %arr)
  unreachable
}
