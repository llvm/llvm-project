; RUN: opt -march=hexagon -hexagon-loop-idiom -S < %s | FileCheck %s
; RUN: opt -march=hexagon -p hexagon-loop-idiom -S < %s | FileCheck %s
; CHECK-LABEL: define void @fred

; Check that this test does not crash.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

%struct.0 = type { [120 x i16], [80 x i16], [80 x i16], [80 x i16], [80 x i16], [80 x i16], [40 x i16], [40 x i16], [40 x i16], [40 x i16], [40 x i16], [40 x i16] }

define void @fred(ptr %demod_state) local_unnamed_addr #0 {
entry:
  br label %for.body309

for.body309:                                      ; preds = %for.body309, %entry
  %max_diff.0300 = phi i16 [ %max_diff.1, %for.body309 ], [ 0, %entry ]
  %arrayidx322.phi = phi ptr [ undef, %entry ], [ %arrayidx322.inc, %for.body309 ]
  %arrayidx331.phi = phi ptr [ undef, %entry ], [ %arrayidx331.inc, %for.body309 ]
  %lag.4299.apmt = phi i32 [ %inc376.apmt, %for.body309 ], [ 0, %entry ]
  %0 = load i16, ptr %arrayidx322.phi, align 2
  %conv323 = sext i16 %0 to i32
  %sub324 = sub nsw i32 0, %conv323
  %ispos258 = icmp sgt i32 %sub324, -1
  %1 = select i1 %ispos258, i32 %sub324, i32 0
  %add326 = add nsw i32 %1, 0
  %2 = load i16, ptr %arrayidx331.phi, align 2
  %conv332 = sext i16 %2 to i32
  %sub333 = sub nsw i32 0, %conv332
  %ispos260 = icmp sgt i32 %sub333, -1
  %3 = select i1 %ispos260, i32 %sub333, i32 undef
  %sub342 = sub nsw i32 0, %conv323
  %ispos262 = icmp sgt i32 %sub342, -1
  %4 = select i1 %ispos262, i32 %sub342, i32 undef
  %sub351 = sub nsw i32 0, %conv332
  %ispos264 = icmp sgt i32 %sub351, -1
  %5 = select i1 %ispos264, i32 %sub351, i32 0
  %sub360 = sub nsw i32 %conv323, %conv332
  %ispos266 = icmp sgt i32 %sub360, -1
  %6 = select i1 %ispos266, i32 %sub360, i32 0
  %add335 = add nsw i32 %add326, %4
  %add344 = add nsw i32 %add335, %3
  %add353 = add i32 %add344, %5
  %add362 = add i32 %add353, %6
  %div363 = sdiv i32 %add362, 6
  %conv364 = trunc i32 %div363 to i16
  %sext268 = shl i32 %div363, 16
  %conv369 = ashr exact i32 %sext268, 16
  %conv370 = sext i16 %max_diff.0300 to i32
  %cmp371 = icmp sgt i32 %conv369, %conv370
  %max_diff.1 = select i1 %cmp371, i16 %conv364, i16 %max_diff.0300
  %inc376.apmt = add nuw nsw i32 %lag.4299.apmt, 1
  %exitcond331 = icmp ne i32 %inc376.apmt, 40
  %arrayidx322.inc = getelementptr i16, ptr %arrayidx322.phi, i32 1
  %arrayidx331.inc = getelementptr i16, ptr %arrayidx331.phi, i32 1
  br i1 %exitcond331, label %for.body309, label %for.end377

for.end377:                                       ; preds = %for.body309
  %max_diff.1.lcssa = phi i16 [ %max_diff.1, %for.body309 ]
  %cmp407 = icmp sgt i16 %max_diff.1.lcssa, 4
  br label %for.body405

for.body405:                                      ; preds = %if.end437, %for.end377
  %arrayidx412 = getelementptr inbounds %struct.0, ptr %demod_state, i32 0, i32 11, i32 undef
  br i1 %cmp407, label %if.then409, label %if.end437

if.then409:                                       ; preds = %for.body405
  %arrayidx416 = getelementptr inbounds [40 x i16], ptr null, i32 0, i32 undef
  %7 = load i16, ptr %arrayidx416, align 2
  %conv417 = sext i16 %7 to i32
  %shl = shl i32 %conv417, 4
  %mul419 = mul nsw i32 %shl, 655
  %add420 = add nsw i32 %mul419, 0
  br label %if.end437

if.end437:                                        ; preds = %if.then409, %for.body405
  %mul431.sink = phi i32 [ %add420, %if.then409 ], [ undef, %for.body405 ]
  %shr432257 = lshr i32 %mul431.sink, 15
  %conv433 = trunc i32 %shr432257 to i16
  store i16 %conv433, ptr %arrayidx412, align 2
  br label %for.body405
}

attributes #0 = { noinline nounwind "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" }
