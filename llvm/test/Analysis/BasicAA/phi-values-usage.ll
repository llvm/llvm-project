; RUN: opt -debug-pass-manager -aa-pipeline=basic-aa -passes='require<phi-values>,memcpyopt,instcombine' -disable-output < %s 2>&1 | FileCheck %s

; Check that phi values is not run when it's not already available, and that
; basicaa is not freed after a pass that preserves CFG, as it preserves CFG.

; CHECK-DAG: Running analysis: PhiValuesAnalysis
; CHECK-DAG: Running analysis: BasicAA
; CHECK-DAG: Running analysis: MemorySSA
; CHECK: Running pass: MemCpyOptPass
; CHECK-NOT: Invalidating analysis
; CHECK: Running pass: InstCombinePass

target datalayout = "p:8:8-n8"

declare void @otherfn(ptr)
declare i32 @__gxx_personality_v0(...)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
@c = external global ptr, align 1

; This function is one where if we didn't free basicaa after memcpyopt then the
; usage of basicaa in instcombine would cause a segfault due to stale phi-values
; results being used.
define void @fn(ptr %this, ptr %ptr, i1 %arg) personality ptr @__gxx_personality_v0 {
entry:
  %arr = alloca [4 x i8], align 8
  br i1 %arg, label %then, label %if

if:
  br label %then

then:
  %phi = phi ptr [ %ptr, %if ], [ null, %entry ]
  store i8 1, ptr %arr, align 8
  %load = load i64, ptr %phi, align 8
  %gep2 = getelementptr inbounds i8, ptr undef, i64 %load
  %gep3 = getelementptr inbounds i8, ptr %gep2, i64 40
  invoke i32 undef(ptr undef)
     to label %invoke unwind label %lpad

invoke:
  unreachable

lpad:
  landingpad { ptr, i32 }
     catch ptr null
  call void @otherfn(ptr nonnull %arr)
  unreachable
}

; When running instcombine after memdep, the basicaa used by instcombine uses
; the phivalues that memdep used. This would then cause a segfault due to
; instcombine deleting a phi whose values had been cached.
define void @fn2(i1 %arg) {
entry:
  %a = alloca i8, align 1
  %0 = load ptr, ptr @c, align 1
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %d.0 = phi ptr [ %0, %entry ], [ null, %for.body ]
  br i1 %arg, label %for.body, label %for.cond.cleanup

for.body:                                         ; preds = %for.cond
  store volatile i8 undef, ptr %a, align 1
  br label %for.cond

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0(i64 1, ptr %a)
  %1 = load ptr, ptr %d.0, align 1
  store ptr %1, ptr @c, align 1
  ret void
}
