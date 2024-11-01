; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

declare void @foo()

define void @test1() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %i.0 = phi i32 [ 2, %entry ], [ %dec, %if.end ]
  switch i32 %i.0, label %if.end [
    i32 0, label %for.end
    i32 1, label %if.then
  ]

if.then:                                          ; preds = %for.cond
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %for.cond, %if.then
  %dec = add nsw i32 %i.0, -1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void

; CHECK-LABEL: @test1
; CHECK: Loop %for.cond: backedge-taken count is 2
; CHECK: Loop %for.cond: constant max backedge-taken count is 2
}
