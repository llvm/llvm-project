; REQUIRES: asserts
; RUN: llc -march=hexagon -enable-pipeliner -o /dev/null < %s
; Test that we do not crash when running CopyToPhi DAG mutation due to
; iterator invalidation.

declare i64 @llvm.hexagon.M2.cmacsc.s0(i64, i32, i32) #0
define dso_local void @foo() local_unnamed_addr #1 {
entry:
  br label %for.body
for.body:                                         ; preds = %for.body, %entry
  %loop_count.0420 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %tmp_rslt4.sroa.0.0415 = phi i64 [ 0, %entry ], [ %phitmp395, %for.body ]
  %tmp_rslt4.sroa.14.0414 = phi i64 [ 0, %entry ], [ %phitmp394, %for.body ]
  %tmp_rslt3.sroa.12.0412 = phi i64 [ 0, %entry ], [ %phitmp391, %for.body ]
  %tmp_rslt3.sroa.0.0.insert.insert = or i64 0, %tmp_rslt3.sroa.12.0412
  %0 = tail call i64 @llvm.hexagon.M2.cmacsc.s0(i64 %tmp_rslt3.sroa.0.0.insert.insert, i32 undef, i32 undef)
  %tmp_rslt4.sroa.0.0.insert.insert = or i64 %tmp_rslt4.sroa.0.0415, %tmp_rslt4.sroa.14.0414
  %1 = tail call i64 @llvm.hexagon.M2.cmacsc.s0(i64 %tmp_rslt4.sroa.0.0.insert.insert, i32 undef, i32 undef)
  %inc = add nuw nsw i32 %loop_count.0420, 1
  %phitmp391 = and i64 %0, -4294967296
  %phitmp394 = and i64 %1, -4294967296
  %phitmp395 = and i64 %1, 4294967295
  %exitcond = icmp eq i32 %inc, 63
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %2 = tail call i64 @llvm.hexagon.M2.cmacsc.s0(i64 %0, i32 undef, i32 undef)
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { "target-features"="-long-calls,-small-data" }
