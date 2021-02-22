; RUN: opt -debug-pass=Executions -phi-values -memcpyopt -instcombine -disable-output < %s -enable-new-pm=0 -enable-memcpyopt-memoryssa=0 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-MEMCPY
; RUN: opt -debug-pass=Executions -memdep -instcombine -disable-output < %s -enable-new-pm=0 2>&1 | FileCheck %s -check-prefix=CHECK
; RUN: opt -debug-pass-manager -aa-pipeline=basic-aa -passes=memcpyopt,instcombine -disable-output -enable-memcpyopt-memoryssa=0 < %s 2>&1 | FileCheck %s -check-prefixes=NPM

; Check that phi values is not run when it's not already available, and that
; basicaa is not freed after a pass that preserves CFG, as it preserves CFG.

; CHECK: Executing Pass 'Phi Values Analysis'
; CHECK: Executing Pass 'Basic Alias Analysis (stateless AA impl)'
; CHECK: Executing Pass 'Memory Dependence Analysis'
; CHECK-MEMCPY: Executing Pass 'MemCpy Optimization'
; CHECK-MEMCPY-DAG: Freeing Pass 'MemCpy Optimization'
; CHECK-DAG: Freeing Pass 'Memory Dependence Analysis'
; CHECK-DAG: Freeing Pass 'Phi Values Analysis'
; CHECK-NOT: Executing Pass 'Phi Values Analysis'
; CHECK-NOT: Executing Pass 'Basic Alias Analysis (stateless AA impl)'
; CHECK: Executing Pass 'Combine redundant instructions'

; NPM-DAG: Running analysis: PhiValuesAnalysis
; NPM-DAG: Running analysis: BasicAA
; NPM-DAG: Running analysis: MemoryDependenceAnalysis
; NPM: Running pass: MemCpyOptPass
; NPM-NOT: Invalidating analysis
; NPM: Running pass: InstCombinePass

target datalayout = "p:8:8-n8"

declare void @otherfn([4 x i8]*)
declare i32 @__gxx_personality_v0(...)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
@c = external global i8*, align 1

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

; When running instcombine after memdep, the basicaa used by instcombine uses
; the phivalues that memdep used. This would then cause a segfault due to
; instcombine deleting a phi whose values had been cached.
define void @fn2() {
entry:
  %a = alloca i8, align 1
  %0 = load i8*, i8** @c, align 1
  %1 = bitcast i8* %0 to i8**
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %d.0 = phi i8** [ %1, %entry ], [ null, %for.body ]
  br i1 undef, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %for.cond
  store volatile i8 undef, i8* %a, align 1
  br label %for.cond

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %a)
  %2 = load i8*, i8** %d.0, align 1
  store i8* %2, i8** @c, align 1
  ret void
}
