; RUN: opt -p hexagon-loop-idiom -S -mtriple hexagon-unknown-elf \
; RUN:   -pass-remarks=hexagon-lir -pass-remarks-missed=hexagon-lir \
; RUN:   %s -o /dev/null 2>&1 | FileCheck %s

;; Test that HexagonLoopIdiomRecognition emits optimization remarks.

;; -- Success: loop converted to memmove --
; CHECK: remark: {{.*}} converted loop to memmove

define void @test_memmove(ptr nocapture %A, ptr nocapture readonly %B, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %src.phi = phi ptr [ %B, %for.body.preheader ], [ %src.inc, %for.body ]
  %dst.phi = phi ptr [ %A, %for.body.preheader ], [ %dst.inc, %for.body ]
  %i = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %val = load i32, ptr %src.phi, align 4
  store i32 %val, ptr %dst.phi, align 4
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp ne i32 %inc, %n
  %src.inc = getelementptr i32, ptr %src.phi, i32 1
  %dst.inc = getelementptr i32, ptr %dst.phi, i32 1
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

;; -- Missed: store value is not a simple load --
; CHECK: remark: {{.*}} store value is not a simple load

define void @test_no_load(ptr nocapture %A, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %dst.phi = phi ptr [ %A, %for.body.preheader ], [ %dst.inc, %for.body ]
  %i = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  store i32 42, ptr %dst.phi, align 4
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp ne i32 %inc, %n
  %dst.inc = getelementptr i32, ptr %dst.phi, i32 1
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

;; -- Missed: non-countable loop --
; CHECK: remark: {{.*}} backedge-taken count is not loop-invariant

define void @test_non_countable(ptr nocapture %A, ptr nocapture readonly %B,
                                ptr nocapture readonly %cond) {
entry:
  br label %for.body

for.body:
  %src.phi = phi ptr [ %B, %entry ], [ %src.inc, %for.body ]
  %dst.phi = phi ptr [ %A, %entry ], [ %dst.inc, %for.body ]
  %val = load i32, ptr %src.phi, align 4
  store i32 %val, ptr %dst.phi, align 4
  %src.inc = getelementptr i32, ptr %src.phi, i32 1
  %dst.inc = getelementptr i32, ptr %dst.phi, i32 1
  %flag = load volatile i1, ptr %cond, align 1
  br i1 %flag, label %for.body, label %for.end

for.end:
  ret void
}
