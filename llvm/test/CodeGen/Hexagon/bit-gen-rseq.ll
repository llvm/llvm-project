; RUN: llc -march=hexagon -disable-hsdr -hexagon-subreg-liveness < %s | FileCheck %s
; Check that we don't generate any bitwise operations.

; CHECK-NOT: = or(
; CHECK-NOT: = and(

target triple = "hexagon"

define i32 @fred(i32* nocapture readonly %p, i32 %n) #0 {
entry:
  %t.sroa.0.048 = load i32, i32* %p, align 4
  %cmp49 = icmp ugt i32 %n, 1
  br i1 %cmp49, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %t.sroa.0.052 = phi i32 [ %t.sroa.0.0, %for.body ], [ %t.sroa.0.048, %entry ]
  %t.sroa.11.051 = phi i64 [ %t.sroa.11.0.extract.shift, %for.body ], [ 0, %entry ]
  %i.050 = phi i32 [ %inc, %for.body ], [ 1, %entry ]
  %t.sroa.0.0.insert.ext = zext i32 %t.sroa.0.052 to i64
  %t.sroa.0.0.insert.insert = or i64 %t.sroa.0.0.insert.ext, %t.sroa.11.051
  %0 = tail call i64 @llvm.hexagon.A2.addp(i64 %t.sroa.0.0.insert.insert, i64 %t.sroa.0.0.insert.insert)
  %t.sroa.11.0.extract.shift = and i64 %0, -4294967296
  %arrayidx4 = getelementptr inbounds i32, i32* %p, i32 %i.050
  %inc = add nuw i32 %i.050, 1
  %t.sroa.0.0 = load i32, i32* %arrayidx4, align 4
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %t.sroa.0.0.lcssa = phi i32 [ %t.sroa.0.048, %entry ], [ %t.sroa.0.0, %for.body ]
  %t.sroa.11.0.lcssa = phi i64 [ 0, %entry ], [ %t.sroa.11.0.extract.shift, %for.body ]
  %t.sroa.0.0.insert.ext17 = zext i32 %t.sroa.0.0.lcssa to i64
  %t.sroa.0.0.insert.insert19 = or i64 %t.sroa.0.0.insert.ext17, %t.sroa.11.0.lcssa
  %1 = tail call i64 @llvm.hexagon.A2.addp(i64 %t.sroa.0.0.insert.insert19, i64 %t.sroa.0.0.insert.insert19)
  %t.sroa.11.0.extract.shift41 = lshr i64 %1, 32
  %t.sroa.11.0.extract.trunc42 = trunc i64 %t.sroa.11.0.extract.shift41 to i32
  ret i32 %t.sroa.11.0.extract.trunc42
}

declare i64 @llvm.hexagon.A2.addp(i64, i64) #1

attributes #0 = { norecurse nounwind readonly }
attributes #1 = { nounwind readnone }
