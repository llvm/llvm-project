; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
;
; Generate loop1 instruction for double loop sequence.

; CHECK: loop1(.LBB{{.}}_{{.}},#100)
; CHECK: loop0(.LBB{{.}}_{{.}},#100)
; CHECK: endloop0
; CHECK: endloop1

define i32 @main() #0 {
entry:
  %array = alloca [100 x i32], align 8
  %doublearray = alloca [100 x [100 x i32]], align 8
  %0 = bitcast [100 x i32]* %array to i8*
  call void @llvm.lifetime.start(i64 400, i8* %0) #1
  %1 = bitcast [100 x [100 x i32]]* %doublearray to i8*
  call void @llvm.lifetime.start(i64 40000, i8* %1) #1
  %arrayidx1 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %doublearray, i32 0, i32 10, i32 10
  %arrayidx2.gep = getelementptr [100 x i32], [100 x i32]* %array, i32 0, i32 0
  br label %for.body

for.body:
  %2 = phi i32 [ undef, %entry ], [ %.pre, %for.body.for.body_crit_edge ]
  %sum.031 = phi i32 [ undef, %entry ], [ %add, %for.body.for.body_crit_edge ]
  %arrayidx2.phi = phi i32* [ %arrayidx2.gep, %entry ], [ %arrayidx2.inc, %for.body.for.body_crit_edge ]
  %i.030 = phi i32 [ 1, %entry ], [ %phitmp, %for.body.for.body_crit_edge ]
  %add = add nsw i32 %2, %sum.031
  %exitcond33 = icmp eq i32 %i.030, 100
  %arrayidx2.inc = getelementptr i32, i32* %arrayidx2.phi, i32 1
  br i1 %exitcond33, label %for.cond7.preheader.preheader, label %for.body.for.body_crit_edge

for.cond7.preheader.preheader:
  br label %for.cond7.preheader

for.body.for.body_crit_edge:
  %.pre = load i32, i32* %arrayidx2.inc, align 4
  %phitmp = add i32 %i.030, 1
  br label %for.body

for.cond7.preheader:
  %i.129 = phi i32 [ %inc16, %for.inc15 ], [ 0, %for.cond7.preheader.preheader ]
  br label %for.body9

for.body9:
  %j.028 = phi i32 [ 0, %for.cond7.preheader ], [ %inc13, %for.body9 ]
  %arrayidx11 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* %doublearray, i32 0, i32 %i.129, i32 %j.028
  store i32 %add, i32* %arrayidx11, align 4
  %inc13 = add nsw i32 %j.028, 1
  %exitcond = icmp eq i32 %inc13, 100
  br i1 %exitcond, label %for.inc15, label %for.body9

for.inc15:
  %inc16 = add nsw i32 %i.129, 1
  %exitcond32 = icmp eq i32 %inc16, 100
  br i1 %exitcond32, label %for.end17, label %for.cond7.preheader

for.end17:
  %3 = load i32, i32* %arrayidx1, align 8
  call void @llvm.lifetime.end(i64 40000, i8* %1) #1
  call void @llvm.lifetime.end(i64 400, i8* %0) #1
  ret i32 %3
}

declare void @llvm.lifetime.start(i64, i8* nocapture) #1

declare void @llvm.lifetime.end(i64, i8* nocapture) #1
