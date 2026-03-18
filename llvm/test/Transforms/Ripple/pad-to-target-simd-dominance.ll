; REQUIRES: hexagon-registered-target
; RUN: opt -passes=ripple -ripple-pad-to-target-simd -disable-output < %s

; This test verifies that the Ripple pass does not generate invalid IR
; (breaking dominance) when padding reduction accumulators that have already
; been naturally padded in an inner loop.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@C = external global ptr

define void @_Z3runv() #0 {
entry.split:
  %0 = load ptr, ptr @C, align 4
  %1 = tail call ptr @llvm.ripple.block.setshape.i32(i32 0, i32 32, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1)
  %2 = tail call i32 @llvm.ripple.block.index.i32(ptr %1, i32 0)
  br label %polly.split_new_and_old306

polly.split_new_and_old306:
  br i1 false, label %polly.loop_if456, label %polly.cond506.thread

for.end27:
  %cmp33 = icmp slt i32 %2, 32
  br i1 %cmp33, label %for.cond36.preheader, label %ripple.par.for.end

for.cond36.preheader:
  %3 = getelementptr i8, ptr null, i32 %2
  br label %for.body39

for.cond.cleanup38:
  store i8 %add47, ptr %3, align 1
  br label %ripple.par.for.end

for.body39:
  %sum34.0198.reg2mem.0 = phi i8 [ 0, %for.cond36.preheader ], [ %add47, %for.body39 ]
  %arrayidx41.phi = phi ptr [ %3, %for.cond36.preheader ], [ null, %for.body39 ]
  %arrayidx43.phi = phi ptr [ %0, %for.cond36.preheader ], [ null, %for.body39 ]
  %j35.0197 = phi i32 [ 0, %for.cond36.preheader ], [ %inc50, %for.body39 ]
  %4 = load i8, ptr %arrayidx41.phi, align 1
  %5 = load i8, ptr %arrayidx43.phi, align 1
  %mul45 = mul i8 %5, %4
  %add47 = add i8 %mul45, %sum34.0198.reg2mem.0
  %inc50 = add nuw nsw i32 %j35.0197, 1
  %exitcond207.not = icmp eq i32 %inc50, 5
  br i1 %exitcond207.not, label %for.cond.cleanup38, label %for.body39

ripple.par.for.end:
  br label %polly.split_new_and_old306

polly.cond506.thread:
  br label %for.end27

polly.loop_if456:
  br label %for.end27
}

declare ptr @llvm.ripple.block.setshape.i32(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare i32 @llvm.ripple.block.index.i32(ptr, i32 immarg) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) "frame-pointer"="all" "no-trapping-math"="true" "polly-optimized" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv73" "target-features"="+hvx-length128b,+hvxv73,+v73,-long-calls" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }