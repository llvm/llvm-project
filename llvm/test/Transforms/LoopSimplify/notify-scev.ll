; RUN: opt -indvars -S %s | FileCheck %s
;
; PR18384: ValueHandleBase::ValueIsDeleted.
;
; Ensure that LoopSimplify calls ScalarEvolution::forgetLoop before
; deleting a block, regardless of whether any values were hoisted out
; of the block.

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

%struct.Params = type { [2 x [4 x [16 x i16]]] }

; Verify that the loop tail is deleted, and we don't crash!
;
; CHECK-LABEL: @t
; CHECK-LABEL: for.cond127.preheader:
; CHECK-NOT: for.cond127:
; CHECK-LABEL: for.body129:
define void @t() {
entry:
  br label %for.body102

for.body102:
  br i1 undef, label %for.cond127.preheader, label %for.inc203

for.cond127.preheader:
  br label %for.body129

for.cond127:
  %cmp128 = icmp slt i32 %inc191, 2
  br i1 %cmp128, label %for.body129, label %for.end192

for.body129:
  %uv.013 = phi i32 [ 0, %for.cond127.preheader ], [ %inc191, %for.cond127 ]
  %idxprom130 = sext i32 %uv.013 to i64
  br i1 undef, label %for.cond135.preheader.lr.ph, label %for.end185

for.cond135.preheader.lr.ph:
  br i1 undef, label %for.cond135.preheader.lr.ph.split.us, label %for.cond135.preheader.lr.ph.split_crit_edge

for.cond135.preheader.lr.ph.split_crit_edge:
  br label %for.cond135.preheader.lr.ph.split

for.cond135.preheader.lr.ph.split.us:
  br label %for.cond135.preheader.us

for.cond135.preheader.us:
  %block_y.09.us = phi i32 [ 0, %for.cond135.preheader.lr.ph.split.us ], [ %add184.us, %for.cond132.us ]
  br i1 true, label %for.cond138.preheader.lr.ph.us, label %for.end178.us

for.end178.us:
  %add184.us = add nsw i32 %block_y.09.us, 4
  br i1 undef, label %for.end185split.us-lcssa.us, label %for.cond132.us

for.end174.us:
  br i1 undef, label %for.cond138.preheader.us, label %for.cond135.for.end178_crit_edge.us

for.inc172.us:
  br i1 undef, label %for.cond142.preheader.us, label %for.end174.us

for.body145.us:
  %arrayidx163.us = getelementptr inbounds %struct.Params, %struct.Params* undef, i64 0, i32 0, i64 %idxprom130, i64 %idxprom146.us
  br i1 undef, label %for.body145.us, label %for.inc172.us

for.cond142.preheader.us:
  %j.04.us = phi i32 [ %block_y.09.us, %for.cond138.preheader.us ], [ undef, %for.inc172.us ]
  %idxprom146.us = sext i32 %j.04.us to i64
  br label %for.body145.us

for.cond138.preheader.us:
  br label %for.cond142.preheader.us

for.cond132.us:
  br i1 undef, label %for.cond135.preheader.us, label %for.cond132.for.end185_crit_edge.us-lcssa.us

for.cond138.preheader.lr.ph.us:
  br label %for.cond138.preheader.us

for.cond135.for.end178_crit_edge.us:
  br label %for.end178.us

for.end185split.us-lcssa.us:
  br label %for.end185split

for.cond132.for.end185_crit_edge.us-lcssa.us:
  br label %for.cond132.for.end185_crit_edge

for.cond135.preheader.lr.ph.split:
  br label %for.end185split

for.end185split:
  br label %for.end185

for.cond132.for.end185_crit_edge:
  br label %for.end185

for.end185:
  %inc191 = add nsw i32 %uv.013, 1
  br i1 false, label %for.end192, label %for.cond127

for.end192:
  br label %for.inc203

for.inc203:
  br label %for.end205

for.end205:
  ret void
}
