; RUN: llc -mtriple=hexagon -pass-remarks=hwloops -pass-remarks-missed=hwloops \
; RUN:   %s -o /dev/null 2>&1 | FileCheck %s

;; Test that HexagonHardwareLoops emits optimization remarks.

;; -- Success: converted loop to hardware loop --
; CHECK: remark: {{.*}} converted loop to hardware loop

define void @test_hwloop(ptr nocapture %a, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i
  store i32 %i, ptr %arrayidx, align 4
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

;; -- Missed: loop contains a call --
; CHECK: remark: {{.*}} loop contains an instruction that prevents hardware loop generation

declare void @bar()

define void @test_call_in_loop(ptr nocapture %a, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %i
  store i32 %i, ptr %arrayidx, align 4
  call void @bar()
  %inc = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
