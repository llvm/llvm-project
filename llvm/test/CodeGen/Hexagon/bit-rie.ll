; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK-LABEL: LBB0{{.*}}if.end
; CHECK: r[[REG:[0-9]+]] = zxth
; CHECK: lsr(r[[REG]],

target triple = "hexagon"

@g0 = external constant [146 x i16], align 8
@g1 = external constant [0 x i16], align 2

define void @fred(ptr nocapture readonly %p0, i16 signext %p1, ptr nocapture %p2, i16 signext %p3, i16 signext %p4, i16 signext %p5) #0 {
entry:
  %conv = sext i16 %p1 to i32
  %0 = tail call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %conv, i32 1)
  %1 = tail call i32 @llvm.hexagon.A2.sath(i32 %0)
  %2 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 3, i32 %1)
  %conv3 = sext i16 %p4 to i32
  %cmp144 = icmp sgt i16 %p4, 0
  br i1 %cmp144, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %arrayidx.phi = phi ptr [ %arrayidx.inc, %for.body ], [ %p0, %entry ]
  %i.0146.apmt = phi i32 [ %inc.apmt, %for.body ], [ 0, %entry ]
  %L_temp1.0145 = phi i32 [ %5, %for.body ], [ 1, %entry ]
  %3 = load i32, ptr %arrayidx.phi, align 4, !tbaa !1
  %4 = tail call i32 @llvm.hexagon.A2.abssat(i32 %3)
  %5 = tail call i32 @llvm.hexagon.A2.max(i32 %L_temp1.0145, i32 %4)
  %inc.apmt = add nuw nsw i32 %i.0146.apmt, 1
  %exitcond151 = icmp eq i32 %inc.apmt, %conv3
  %arrayidx.inc = getelementptr i32, ptr %arrayidx.phi, i32 1
  br i1 %exitcond151, label %for.end, label %for.body, !llvm.loop !5

for.end:                                          ; preds = %for.body, %entry
  %L_temp1.0.lcssa = phi i32 [ 1, %entry ], [ %5, %for.body ]
  %6 = tail call i32 @llvm.hexagon.S2.clbnorm(i32 %L_temp1.0.lcssa)
  %arrayidx6 = getelementptr inbounds [146 x i16], ptr @g0, i32 0, i32 %conv3
  %7 = load i16, ptr %arrayidx6, align 2, !tbaa !7
  %conv7 = sext i16 %7 to i32
  %8 = tail call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %6, i32 %conv7)
  br i1 %cmp144, label %for.body14.lr.ph, label %for.end29

for.body14.lr.ph:                                 ; preds = %for.end
  %sext132 = shl i32 %8, 16
  %conv17 = ashr exact i32 %sext132, 16
  br label %for.body14

for.body14:                                       ; preds = %for.body14, %for.body14.lr.ph
  %arrayidx16.phi = phi ptr [ %p0, %for.body14.lr.ph ], [ %arrayidx16.inc, %for.body14 ]
  %i.1143.apmt = phi i32 [ 0, %for.body14.lr.ph ], [ %inc28.apmt, %for.body14 ]
  %L_temp.0142 = phi i32 [ 0, %for.body14.lr.ph ], [ %12, %for.body14 ]
  %9 = load i32, ptr %arrayidx16.phi, align 4, !tbaa !1
  %10 = tail call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %9, i32 %conv17)
  %11 = tail call i32 @llvm.hexagon.A2.asrh(i32 %10)
  %sext133 = shl i32 %11, 16
  %conv23 = ashr exact i32 %sext133, 16
  %12 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32 %L_temp.0142, i32 %conv23, i32 %conv23)
  %inc28.apmt = add nuw nsw i32 %i.1143.apmt, 1
  %exitcond = icmp eq i32 %inc28.apmt, %conv3
  %arrayidx16.inc = getelementptr i32, ptr %arrayidx16.phi, i32 1
  br i1 %exitcond, label %for.end29, label %for.body14

for.end29:                                        ; preds = %for.body14, %for.end
  %L_temp.0.lcssa = phi i32 [ 0, %for.end ], [ %12, %for.body14 ]
  %13 = tail call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %conv3, i32 1)
  %cmp31 = icmp sgt i32 %13, 0
  br i1 %cmp31, label %if.then, label %if.end

if.then:                                          ; preds = %for.end29
  %arrayidx34 = getelementptr inbounds [0 x i16], ptr @g1, i32 0, i32 %conv3
  %14 = load i16, ptr %arrayidx34, align 2, !tbaa !7
  %cmp.i = icmp eq i32 %L_temp.0.lcssa, -2147483648
  %cmp1.i = icmp eq i16 %14, -32768
  %or.cond.i = and i1 %cmp.i, %cmp1.i
  br i1 %or.cond.i, label %if.end, label %if.else.i

if.else.i:                                        ; preds = %if.then
  %conv3.i = sext i16 %14 to i32
  %15 = tail call i32 @llvm.hexagon.M2.hmmpyl.s1(i32 %L_temp.0.lcssa, i32 %conv3.i) #2
  %16 = tail call i64 @llvm.hexagon.M2.mpyd.ll.s1(i32 %conv3.i, i32 %L_temp.0.lcssa) #2
  %conv5.i = trunc i64 %16 to i32
  %phitmp = and i32 %conv5.i, 65535
  br label %if.end

if.end:                                           ; preds = %if.else.i, %if.then, %for.end29
  %L_temp.2 = phi i32 [ %L_temp.0.lcssa, %for.end29 ], [ %15, %if.else.i ], [ 2147483647, %if.then ]
  %lsb.0 = phi i32 [ 0, %for.end29 ], [ %phitmp, %if.else.i ], [ 65535, %if.then ]
  %sext = shl i32 %8, 16
  %conv35 = ashr exact i32 %sext, 16
  %17 = tail call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %conv35, i32 16)
  %18 = tail call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %17, i32 1)
  %19 = tail call i32 @llvm.hexagon.A2.sath(i32 %18)
  %20 = tail call i32 @llvm.hexagon.S2.clbnorm(i32 %L_temp.2)
  %sext123 = shl i32 %20, 16
  %conv38 = ashr exact i32 %sext123, 16
  %sext124 = shl i32 %19, 16
  %conv39 = ashr exact i32 %sext124, 16
  %21 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %conv38, i32 %conv39)
  %22 = tail call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %L_temp.2, i32 %conv38)
  %23 = tail call i32 @llvm.hexagon.A2.zxth(i32 %lsb.0)
  %24 = tail call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 16, i32 %conv38)
  %25 = tail call i32 @llvm.hexagon.S2.lsr.r.r(i32 %23, i32 %24)
  %sext125 = shl i32 %25, 16
  %conv45 = ashr exact i32 %sext125, 16
  %26 = tail call i32 @llvm.hexagon.A2.addsat(i32 %22, i32 %conv45)
  %sext126 = shl i32 %2, 16
  %conv46 = ashr exact i32 %sext126, 16
  %sext127 = shl i32 %21, 16
  %conv47 = ashr exact i32 %sext127, 16
  %27 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %conv46, i32 %conv47)
  %sext128 = shl i32 %27, 16
  %conv49 = ashr exact i32 %sext128, 16
  %cmp50 = icmp sgt i32 %sext128, 327679
  %tobool = icmp eq i16 %p5, 0
  %or.cond = or i1 %tobool, %cmp50
  br i1 %or.cond, label %if.else68, label %if.then53

if.then53:                                        ; preds = %if.end
  %28 = tail call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %conv49, i32 1)
  %29 = tail call i32 @llvm.hexagon.A2.sath(i32 %28)
  %30 = tail call i32 @llvm.hexagon.A2.subsat(i32 %26, i32 1276901417)
  %cmp56 = icmp slt i32 %30, 0
  br i1 %cmp56, label %if.then58, label %if.else

if.then58:                                        ; preds = %if.then53
  %sext131 = shl i32 %29, 16
  %conv59 = ashr exact i32 %sext131, 16
  %31 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %conv59, i32 2)
  br label %if.end80

if.else:                                          ; preds = %if.then53
  %32 = tail call i32 @llvm.hexagon.A2.subsat(i32 %26, i32 1805811301)
  %cmp61 = icmp slt i32 %32, 0
  br i1 %cmp61, label %if.then63, label %if.end80

if.then63:                                        ; preds = %if.else
  %sext130 = shl i32 %29, 16
  %conv64 = ashr exact i32 %sext130, 16
  %33 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %conv64, i32 1)
  br label %if.end80

if.else68:                                        ; preds = %if.end
  %34 = tail call i32 @llvm.hexagon.A2.subsat(i32 %26, i32 1518500250)
  %cmp69 = icmp slt i32 %34, 0
  br i1 %cmp69, label %if.then71, label %if.end74

if.then71:                                        ; preds = %if.else68
  %35 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %conv49, i32 1)
  br label %if.end74

if.end74:                                         ; preds = %if.then71, %if.else68
  %m.0.in = phi i32 [ %35, %if.then71 ], [ %27, %if.else68 ]
  br i1 %tobool, label %if.end80, label %if.then76

if.then76:                                        ; preds = %if.end74
  %sext129 = shl i32 %m.0.in, 16
  %conv77 = ashr exact i32 %sext129, 16
  %36 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %conv77, i32 5)
  br label %if.end80

if.end80:                                         ; preds = %if.end74, %if.then76, %if.then58, %if.then63, %if.else
  %m.1.in = phi i32 [ %31, %if.then58 ], [ %33, %if.then63 ], [ %29, %if.else ], [ %36, %if.then76 ], [ %m.0.in, %if.end74 ]
  %m.1 = trunc i32 %m.1.in to i16
  %cmp.i135 = icmp slt i16 %m.1, 0
  %var_out.0.i136 = select i1 %cmp.i135, i16 0, i16 %m.1
  %conv81 = sext i16 %p3 to i32
  %37 = tail call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %conv81, i32 1)
  %conv82 = trunc i32 %37 to i16
  %cmp.i134 = icmp sgt i16 %var_out.0.i136, %conv82
  %var_out.0.i = select i1 %cmp.i134, i16 %conv82, i16 %var_out.0.i136
  store i16 %var_out.0.i, ptr %p2, align 2, !tbaa !7
  ret void
}

declare i32 @llvm.hexagon.A2.abssat(i32) #2
declare i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32, i32) #2
declare i32 @llvm.hexagon.A2.addsat(i32, i32) #2
declare i32 @llvm.hexagon.A2.asrh(i32) #2
declare i32 @llvm.hexagon.A2.max(i32, i32) #2
declare i32 @llvm.hexagon.A2.sath(i32) #2
declare i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32, i32) #2
declare i32 @llvm.hexagon.A2.subsat(i32, i32) #2
declare i32 @llvm.hexagon.A2.zxth(i32) #2
declare i32 @llvm.hexagon.M2.hmmpyl.s1(i32, i32) #2
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32, i32, i32) #2
declare i32 @llvm.hexagon.S2.asl.r.r.sat(i32, i32) #2
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #2
declare i32 @llvm.hexagon.S2.clbnorm(i32) #2
declare i32 @llvm.hexagon.S2.lsr.r.r(i32, i32) #2
declare i64 @llvm.hexagon.M2.mpyd.ll.s1(i32, i32) #2
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone }


!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.threadify", i32 81508608}
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !3, i64 0}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.threadify", i32 1441813}
!11 = distinct !{!11, !10}
