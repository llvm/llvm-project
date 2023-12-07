; RUN: opt %loadPolly -polly-detect -pass-remarks-missed="polly-detect" -disable-output < %s 2>&1| FileCheck %s

; CHECK: remark: <unknown>:0:0: Irreducible region encountered in control flow.

define void @hoge(ptr %arg)  {
bb1:
  br i1 false, label %bb2, label %bb3

bb2:
  br i1 false, label %bb4, label %bb5

bb4:
  br i1 false, label %bb3, label %bb5

bb5:
  br label %bb6

bb6:
  br label %bb4

bb3:
  ret void
}

