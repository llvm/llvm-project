; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -instsimplify -S | FileCheck %s --check-prefixes=CHECK,LV
; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -prefer-predicate-over-epilogue=predicate-dont-vectorize -instsimplify -S | FileCheck %s --check-prefixes=CHECK,TFALWAYS
; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue -instsimplify -S | FileCheck %s --check-prefixes=CHECK,TFFALLBACK

target triple = "aarch64-unknown-linux-gnu"

; A call whose argument must be widened. We check that tail folding uses the
; primary mask, and that without tail folding we synthesize an all-true mask.
define void @test_widen(i64* noalias %a, i64* readnone %b) #4 {
; CHECK-LABEL: @test_widen(
; LV-NOT: call <vscale x 2 x i64> @foo_vector
; TFALWAYS-NOT: vector.body
; TFALWAYS-NOT: call <vscale x 2 x i64> @foo_vector
; TFFALLBACK-NOT: call <vscale x 2 x i64> @foo_vector
; CHECK: ret void
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %gep = getelementptr i64, i64* %b, i64 %indvars.iv
  %load = load i64, i64* %gep
  %call = call i64 @foo(i64 %load) #1
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %indvars.iv
  store i64 %call, i64* %arrayidx
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; Check that a simple conditional call can be vectorized.
define void @test_if_then(i64* noalias %a, i64* readnone %b) #4 {
; CHECK-LABEL: @test_if_then(
; LV-NOT: call <vscale x 2 x i64> @foo_vector
; TFALWAYS-NOT: call <vscale x 2 x i64> @foo_vector
; TFFALLBACK-NOT: call <vscale x 2 x i64> @foo_vector
; CHECK: ret void
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %indvars.iv
  %0 = load i64, i64* %arrayidx, align 8
  %cmp = icmp ugt i64 %0, 50
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %1 = call i64 @foo(i64 %0) #1
  br label %if.end

if.end:
  %2 = phi i64 [%1, %if.then], [0, %for.body]
  %arrayidx1 = getelementptr inbounds i64, i64* %b, i64 %indvars.iv
  store i64 %2, i64* %arrayidx1, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; This checks the ability to handle masking of an if-then-else CFG with
; calls inside the conditional blocks. Although one of the calls has a
; uniform parameter and the metadata lists a uniform variant, right now
; we just see a splat of the parameter instead. More work needed.
define void @test_widen_if_then_else(i64* noalias %a, i64* readnone %b) #4 {
; CHECK-LABEL: @test_widen_if_then_else
; LV-NOT: call <vscale x 2 x i64> @foo_vector
; LV-NOT: call <vscale x 2 x i64> @foo_uniform
; TFALWAYS-NOT: call <vscale x 2 x i64> @foo_vector
; TFALWAYS-NOT: call <vscale x 2 x i64> @foo_uniform
; TFFALLBACK-NOT: call <vscale x 2 x i64> @foo_vector
; TFFALLBACK-NOT: call <vscale x 2 x i64> @foo_uniform
; CHECK: ret void
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %indvars.iv
  %0 = load i64, i64* %arrayidx, align 8
  %cmp = icmp ugt i64 %0, 50
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %1 = call i64 @foo(i64 %0) #0
  br label %if.end

if.else:
  %2 = call i64 @foo(i64 0) #0
  br label %if.end

if.end:
  %3 = phi i64 [%1, %if.then], [%2, %if.else]
  %arrayidx1 = getelementptr inbounds i64, i64* %b, i64 %indvars.iv
  store i64 %3, i64* %arrayidx1, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; A call whose argument must be widened, where the vector variant does not have
; a mask. Forcing tail folding results in no vectorized call, whereas an
; unpredicated body with scalar tail can use the unmasked variant.
define void @test_widen_nomask(i64* noalias %a, i64* readnone %b) #4 {
; CHECK-LABEL: @test_widen_nomask(
; LV: call <vscale x 2 x i64> @foo_vector_nomask
; TFALWAYS-NOT: vector.body
; TFALWAYS-NOT: call <vscale x 2 x i64> @foo_vector_nomask
; TFFALLBACK: call <vscale x 2 x i64> @foo_vector_nomask
; CHECK: ret void
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %gep = getelementptr i64, i64* %b, i64 %indvars.iv
  %load = load i64, i64* %gep
  %call = call i64 @foo(i64 %load) #2
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %indvars.iv
  store i64 %call, i64* %arrayidx
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; If both masked and unmasked options are present, we expect to see tail folding
; use the masked version and unpredicated body with scalar tail use the unmasked
; version.
define void @test_widen_optmask(i64* noalias %a, i64* readnone %b) #4 {
; CHECK-LABEL: @test_widen_optmask(
; LV: call <vscale x 2 x i64> @foo_vector_nomask
; TFALWAYS-NOT: vector.body
; TFALWAYS-NOT: call <vscale x 2 x i64> @foo_vector
; TFFALLBACK: call <vscale x 2 x i64> @foo_vector_nomask
; CHECK: ret void
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %gep = getelementptr i64, i64* %b, i64 %indvars.iv
  %load = load i64, i64* %gep
  %call = call i64 @foo(i64 %load) #3
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %indvars.iv
  store i64 %call, i64* %arrayidx
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

declare i64 @foo(i64)

; vector variants of foo
declare <vscale x 2 x i64> @foo_uniform(i64, <vscale x 2 x i1>)
declare <vscale x 2 x i64> @foo_vector(<vscale x 2 x i64>, <vscale x 2 x i1>)
declare <vscale x 2 x i64> @foo_vector_nomask(<vscale x 2 x i64>)

attributes #0 = { nounwind "vector-function-abi-variant"="_ZGV_LLVM_Mxv_foo(foo_vector),_ZGV_LLVM_Mxu_foo(foo_uniform)" }
attributes #1 = { nounwind "vector-function-abi-variant"="_ZGV_LLVM_Mxv_foo(foo_vector)" }
attributes #2 = { nounwind "vector-function-abi-variant"="_ZGV_LLVM_Nxv_foo(foo_vector_nomask)" }
attributes #3 = { nounwind "vector-function-abi-variant"="_ZGV_LLVM_Nxv_foo(foo_vector_nomask),_ZGV_LLVM_Mxv_foo(foo_vector)" }
attributes #4 = { "target-features"="+sve" vscale_range(2,16) "no-trapping-math"="false" }
