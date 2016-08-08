; RUN: llc -march=hexagon -mcpu=hexagonv5 -enable-pipeliner \
; RUN:     -pipeliner-max-stages=2 < %s | FileCheck %s

@A = global [8 x i32] [i32 4, i32 -3, i32 5, i32 -2, i32 -1, i32 2, i32 6, i32 -2], align 8

define i32 @test(i32 %Left, i32 %Right) {
entry:
  %add = add nsw i32 %Right, %Left
  %div = sdiv i32 %add, 2
  %cmp9 = icmp slt i32 %div, %Left
  br i1 %cmp9, label %for.end, label %for.body.preheader

for.body.preheader:
  br label %for.body

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: [[REG1:(r[0-9]+)]] = max(r{{[0-9]+}}, [[REG1]])
; CHECK: [[REG0:(r[0-9]+)]] = add([[REG2:(r[0-9]+)]], [[REG0]])
; CHECK: [[REG2]] = memw
; CHECK: endloop0

for.body:
  %MaxLeftBorderSum.012 = phi i32 [ %MaxLeftBorderSum.1, %for.body ], [ 0, %for.body.preheader ]
  %i.011 = phi i32 [ %dec, %for.body ], [ %div, %for.body.preheader ]
  %LeftBorderSum.010 = phi i32 [ %add1, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* @A, i32 0, i32 %i.011
  %0 = load i32, i32* %arrayidx, align 4
  %add1 = add nsw i32 %0, %LeftBorderSum.010
  %cmp2 = icmp sgt i32 %add1, %MaxLeftBorderSum.012
  %MaxLeftBorderSum.1 = select i1 %cmp2, i32 %add1, i32 %MaxLeftBorderSum.012
  %dec = add nsw i32 %i.011, -1
  %cmp = icmp slt i32 %dec, %Left
  br i1 %cmp, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  %MaxLeftBorderSum.0.lcssa = phi i32 [ 0, %entry ], [ %MaxLeftBorderSum.1, %for.end.loopexit ]
  ret i32 %MaxLeftBorderSum.0.lcssa
}
