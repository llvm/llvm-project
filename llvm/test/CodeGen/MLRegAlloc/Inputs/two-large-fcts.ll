; This is a copy of test/CodeGen/X86/ragreedy-hoist-spill.ll. It generates
; sufficiently interesting differences between the default eviction heuristic
; and the test ML policy: different eviction choices, and different reward.
;
;
%struct.TMP.1 = type { %struct.TMP.2*, %struct.TMP.2*, [1024 x i8] }
%struct.TMP.2 = type { i8*, i32, i32, i16, i16, %struct.TMP.3, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.TMP.3, %struct.TMP.4*, i32, [3 x i8], [1 x i8], %struct.TMP.3, i32, i64 }
%struct.TMP.4 = type opaque
%struct.TMP.3 = type { i8*, i32 }

@syBuf = external global [16 x %struct.TMP.1], align 16
@syHistory = external global [8192 x i8], align 16
@SyFgets.yank = external global [512 x i8], align 16
@syCTRO = external global i32, align 4

define i8* @SyFgets(i8* %line, i64 %length, i64 %fid) {
entry:
  %sub.ptr.rhs.cast646 = ptrtoint i8* %line to i64
  %old = alloca [512 x i8], align 16
  %0 = getelementptr inbounds [512 x i8], [512 x i8]* %old, i64 0, i64 0
  switch i64 %fid, label %if.then [
    i64 2, label %if.end
    i64 0, label %if.end
  ]

if.then:
  br label %cleanup

if.end:
  switch i64 undef, label %if.end25 [
    i64 0, label %if.then4
    i64 1, label %if.end25
  ]

if.then4:
  br i1 undef, label %SyTime.exit, label %if.then.i

if.then.i:
  unreachable

SyTime.exit:
  br i1 undef, label %SyTime.exit2681, label %if.then.i2673

if.then.i2673:
  unreachable

SyTime.exit2681:
  br label %cleanup

land.lhs.true14:
  unreachable

if.end25:
  br i1 undef, label %SyTime.exit2720, label %if.then.i2712

if.then.i2712:
  unreachable

SyTime.exit2720:
  %add.ptr = getelementptr [512 x i8], [512 x i8]* %old, i64 0, i64 512
  %cmp293427 = icmp ult i8* %0, %add.ptr
  br i1 %cmp293427, label %for.body.lr.ph, label %while.body.preheader

for.body.lr.ph:
  call void @llvm.memset.p0i8.i64(i8* align 16 undef, i8 32, i64 512, i1 false)
  br label %while.body.preheader

while.body.preheader:
  %add.ptr1603 = getelementptr [512 x i8], [512 x i8]* null, i64 0, i64 512
  %echo.i3101 = getelementptr [16 x %struct.TMP.1], [16 x %struct.TMP.1]* @syBuf, i64 0, i64 %fid, i32 1
  %1 = xor i64 %sub.ptr.rhs.cast646, -1
  br label %do.body

do.body:
  %ch2.0 = phi i32 [ 0, %while.body.preheader ], [ %ch.12.ch2.12, %do.body ]
  %rep.0 = phi i32 [ 1, %while.body.preheader ], [ %rep.6, %do.body ]
  store i32 0, i32* @syCTRO, align 4, !tbaa !1
  %ch.0.ch2.0 = select i1 undef, i32 14, i32 %ch2.0
  %ch2.2 = select i1 undef, i32 0, i32 %ch.0.ch2.0
  %ch.2.ch2.2 = select i1 undef, i32 0, i32 %ch2.2
  %ch2.4 = select i1 undef, i32 278, i32 %ch.2.ch2.2
  %ch2.5 = select i1 undef, i32 0, i32 %ch2.4
  %rep.2 = select i1 undef, i32 undef, i32 %rep.0
  %ch.5.ch2.5 = select i1 undef, i32 undef, i32 %ch2.5
  %ch2.7 = select i1 undef, i32 0, i32 %ch.5.ch2.5
  %rep.3 = select i1 undef, i32 undef, i32 %rep.2
  %ch.7.ch2.7 = select i1 false, i32 0, i32 %ch2.7
  %mul98.rep.3 = select i1 false, i32 0, i32 %rep.3
  %ch2.9 = select i1 undef, i32 undef, i32 %ch.7.ch2.7
  %rep.5 = select i1 undef, i32 undef, i32 %mul98.rep.3
  %ch2.10 = select i1 false, i32 undef, i32 %ch2.9
  %rep.6 = select i1 false, i32 undef, i32 %rep.5
  %isdigittmp = add i32 %ch2.10, -48
  %isdigit = icmp ult i32 %isdigittmp, 10
  %cmp119 = icmp eq i32 undef, 22
  %or.cond1875 = and i1 %isdigit, %cmp119
  %ch.10.ch2.10 = select i1 %or.cond1875, i32 undef, i32 %ch2.10
  %.ch.10 = select i1 %or.cond1875, i32 0, i32 undef
  %ch2.12 = select i1 undef, i32 %.ch.10, i32 %ch.10.ch2.10
  %ch.12 = select i1 undef, i32 0, i32 %.ch.10
  %ch.12.ch2.12 = select i1 false, i32 %ch.12, i32 %ch2.12
  %.ch.12 = select i1 false, i32 0, i32 %ch.12
  %cmp147 = icmp eq i32 %.ch.12, 0
  br i1 %cmp147, label %do.body, label %do.end

do.end:
  %cmp164 = icmp eq i32 %ch.12.ch2.12, 21
  %mul167 = shl i32 %rep.6, 2
  %rep.8 = select i1 %cmp164, i32 %mul167, i32 %rep.6
  %..ch.19 = select i1 false, i32 2, i32 0
  br i1 undef, label %while.body200, label %while.end1465

while.body200:
  %dec3386.in = phi i32 [ %dec3386, %while.cond197.backedge ], [ %rep.8, %do.end ]
  %oldc.13384 = phi i32 [ %oldc.1.be, %while.cond197.backedge ], [ 0, %do.end ]
  %ch.213379 = phi i32 [ %last.1.be, %while.cond197.backedge ], [ %..ch.19, %do.end ]
  %last.13371 = phi i32 [ %last.1.be, %while.cond197.backedge ], [ 0, %do.end ]
  %dec3386 = add i32 %dec3386.in, -1
  switch i32 %ch.213379, label %sw.default [
    i32 1, label %while.cond201.preheader
    i32 322, label %sw.bb206
    i32 354, label %sw.bb206
    i32 2, label %sw.bb243
    i32 364, label %sw.bb1077
    i32 326, label %sw.bb256
    i32 358, label %sw.bb256
    i32 341, label %sw.bb979
    i32 323, label %while.cond1037.preheader
    i32 373, label %sw.bb979
    i32 4, label %if.then1477
    i32 332, label %sw.bb1077
    i32 11, label %for.cond357
    i32 355, label %while.cond1037.preheader
    i32 324, label %sw.bb474
    i32 356, label %sw.bb474
    i32 20, label %sw.bb566
    i32 -1, label %while.cond197.backedge
    i32 268, label %sw.bb1134
    i32 16, label %while.cond635.preheader
    i32 18, label %sw.bb956
    i32 316, label %while.cond864
  ]

while.cond1037.preheader:
  %cmp10393273 = icmp eq i8 undef, 0
  br i1 %cmp10393273, label %if.end1070, label %land.rhs1041

while.cond635.preheader:
  br i1 undef, label %for.body643.us, label %while.cond661

for.body643.us:
  br label %for.body643.us

while.cond201.preheader:
  %umax = select i1 false, i64 undef, i64 %1
  %2 = xor i64 %umax, -1
  %3 = inttoptr i64 %2 to i8*
  br label %while.cond197.backedge

sw.bb206:
  br label %while.cond197.backedge

sw.bb243:
  br label %while.cond197.backedge

sw.bb256:
  br label %while.cond197.backedge

while.cond197.backedge:
  %last.1.be = phi i32 [ %ch.213379, %sw.default ], [ -1, %while.body200 ], [ %ch.213379, %sw.bb1077 ], [ %ch.213379, %sw.bb979 ], [ 18, %sw.bb956 ], [ 20, %sw.bb566 ], [ %ch.213379, %for.end552 ], [ %ch.213379, %sw.bb256 ], [ 2, %sw.bb243 ], [ 1, %while.cond201.preheader ], [ 268, %for.cond1145.preheader ], [ %ch.213379, %sw.bb206 ]
  %oldc.1.be = phi i32 [ %oldc.13384, %sw.default ], [ %oldc.13384, %while.body200 ], [ %oldc.13384, %sw.bb1077 ], [ %oldc.13384, %sw.bb979 ], [ %oldc.13384, %sw.bb956 ], [ %oldc.13384, %sw.bb566 ], [ %oldc.13384, %for.end552 ], [ %oldc.13384, %sw.bb256 ], [ %oldc.13384, %sw.bb243 ], [ %oldc.13384, %while.cond201.preheader ], [ 0, %for.cond1145.preheader ], [ %oldc.13384, %sw.bb206 ]
  %cmp198 = icmp sgt i32 %dec3386, 0
  br i1 %cmp198, label %while.body200, label %while.end1465

for.cond357:
  br label %for.cond357

sw.bb474:
  ; spill is hoisted here. Although loop depth1 is even hotter than loop depth2, sw.bb474 is still cold.
  %cmp476 = icmp eq i8 undef, 0
  br i1 %cmp476, label %if.end517, label %do.body479.preheader

do.body479.preheader:
  %cmp4833314 = icmp eq i8 undef, 0
  br i1 %cmp4833314, label %if.end517, label %land.rhs485

land.rhs485:
  %incdec.ptr4803316 = phi i8* [ %incdec.ptr480, %do.body479.backedge.land.rhs485_crit_edge ], [ undef, %do.body479.preheader ]
  %isascii.i.i27763151 = icmp sgt i8 undef, -1
  br i1 %isascii.i.i27763151, label %cond.true.i.i2780, label %cond.false.i.i2782

cond.true.i.i2780:
  br i1 undef, label %land.lhs.true490, label %lor.rhs500

cond.false.i.i2782:
  unreachable

land.lhs.true490:
  br i1 false, label %lor.rhs500, label %do.body479.backedge

lor.rhs500:
  ; Make sure spill is hoisted to a cold preheader in outside loop.
  %call3.i.i2792 = call i32 @__maskrune(i32 undef, i64 256)
  br i1 undef, label %land.lhs.true504, label %do.body479.backedge

land.lhs.true504:
  br i1 undef, label %do.body479.backedge, label %if.end517

do.body479.backedge:
  %incdec.ptr480 = getelementptr i8, i8* %incdec.ptr4803316, i64 1
  %cmp483 = icmp eq i8 undef, 0
  br i1 %cmp483, label %if.end517, label %do.body479.backedge.land.rhs485_crit_edge

do.body479.backedge.land.rhs485_crit_edge:
  br label %land.rhs485

if.end517:
  %q.4 = phi i8* [ undef, %sw.bb474 ], [ undef, %do.body479.preheader ], [ %incdec.ptr480, %do.body479.backedge ], [ %incdec.ptr4803316, %land.lhs.true504 ]
  switch i32 %last.13371, label %if.then532 [
    i32 383, label %for.cond534
    i32 356, label %for.cond534
    i32 324, label %for.cond534
    i32 24, label %for.cond534
    i32 11, label %for.cond534
  ]

if.then532:
  store i8 0, i8* getelementptr inbounds ([512 x i8], [512 x i8]* @SyFgets.yank, i64 0, i64 0), align 16, !tbaa !5
  br label %for.cond534

for.cond534:
  %cmp536 = icmp eq i8 undef, 0
  br i1 %cmp536, label %for.cond542.preheader, label %for.cond534

for.cond542.preheader:
  br i1 undef, label %for.body545, label %for.end552

for.body545:
  br i1 undef, label %for.end552, label %for.body545

for.end552:
  %s.2.lcssa = phi i8* [ undef, %for.cond542.preheader ], [ %q.4, %for.body545 ]
  %sub.ptr.lhs.cast553 = ptrtoint i8* %s.2.lcssa to i64
  %sub.ptr.sub555 = sub i64 %sub.ptr.lhs.cast553, 0
  %arrayidx556 = getelementptr i8, i8* null, i64 %sub.ptr.sub555
  store i8 0, i8* %arrayidx556, align 1, !tbaa !5
  br label %while.cond197.backedge

sw.bb566:
  br label %while.cond197.backedge

while.cond661:
  br label %while.cond661

while.cond864:
  br label %while.cond864

sw.bb956:
  br i1 undef, label %if.then959, label %while.cond197.backedge

if.then959:
  br label %while.cond962

while.cond962:
  br label %while.cond962

sw.bb979:
  br label %while.cond197.backedge

land.rhs1041:
  unreachable

if.end1070:
  br label %sw.bb1077

sw.bb1077:
  br label %while.cond197.backedge

sw.bb1134:
  br i1 false, label %for.body1139, label %for.cond1145.preheader

for.cond1145.preheader:
  br i1 %cmp293427, label %for.body1150.lr.ph, label %while.cond197.backedge

for.body1150.lr.ph:
  unreachable

for.body1139:
  unreachable

sw.default:
  br label %while.cond197.backedge

while.end1465:
  %oldc.1.lcssa = phi i32 [ 0, %do.end ], [ %oldc.1.be, %while.cond197.backedge ]
  %ch.21.lcssa = phi i32 [ %..ch.19, %do.end ], [ %last.1.be, %while.cond197.backedge ]
  switch i32 %ch.21.lcssa, label %for.cond1480.preheader [
    i32 -1, label %if.then1477
    i32 15, label %if.then1477
    i32 13, label %if.then1477
    i32 10, label %if.then1477
  ]

for.cond1480.preheader:
  br i1 undef, label %for.body1606.lr.ph, label %for.end1609

if.then1477:
  %p.1.lcssa3539 = phi i8* [ null, %while.end1465 ], [ null, %while.end1465 ], [ null, %while.end1465 ], [ null, %while.end1465 ], [ %line, %while.body200 ]
  %call1.i3057 = call i64 @"\01_write"(i32 undef, i8* undef, i64 1)
  %sub.ptr.lhs.cast1717 = ptrtoint i8* %p.1.lcssa3539 to i64
  %sub.ptr.sub1719 = sub i64 %sub.ptr.lhs.cast1717, %sub.ptr.rhs.cast646
  %idx.neg1727 = sub i64 0, %sub.ptr.sub1719
  br label %for.body1723

for.body1606.lr.ph:
  br label %for.end1609

for.end1609:
  br i1 undef, label %for.cond1659.preheader, label %land.lhs.true1614

land.lhs.true1614:
  br label %for.cond1659.preheader

for.cond1659.preheader:
  %cmp16623414 = icmp ult i8* undef, %add.ptr1603
  br i1 %cmp16623414, label %for.body1664.lr.ph, label %while.body1703.lr.ph

for.body1664.lr.ph:
  %cmp16773405 = icmp slt i64 undef, undef
  br i1 %cmp16773405, label %while.body1679, label %while.cond1683.preheader

while.body1703.lr.ph:
  unreachable

while.cond1683.preheader:
  br i1 undef, label %while.body1691, label %while.end1693

while.body1679:
  %oldc.43406 = phi i32 [ %inc, %syEchoch.exit3070 ], [ %oldc.1.lcssa, %for.body1664.lr.ph ]
  %4 = load %struct.TMP.2*, %struct.TMP.2** %echo.i3101, align 8, !tbaa !6
  %call.i3062 = call i32 @fileno(%struct.TMP.2* %4)
  br i1 undef, label %if.then.i3069, label %syEchoch.exit3070

if.then.i3069:
  br label %syEchoch.exit3070

syEchoch.exit3070:
  %inc = add i32 %oldc.43406, 1
  %conv1672 = sext i32 %inc to i64
  %cmp1677 = icmp slt i64 %conv1672, undef
  br i1 %cmp1677, label %while.body1679, label %while.cond1683.preheader

while.body1691:
  unreachable

while.end1693:
  unreachable

for.body1723:
  %q.303203 = phi i8* [ getelementptr inbounds ([8192 x i8], [8192 x i8]* @syHistory, i64 0, i64 8189), %if.then1477 ], [ %incdec.ptr1730, %for.body1723 ]
  %add.ptr1728 = getelementptr i8, i8* %q.303203, i64 %idx.neg1727
  %5 = load i8, i8* %add.ptr1728, align 1, !tbaa !5
  %incdec.ptr1730 = getelementptr i8, i8* %q.303203, i64 -1
  br label %for.body1723

cleanup:
  ret i8* undef
}


define i8* @SyFgetsCopy(i8* %line, i64 %length, i64 %fid) {
entry:
  %sub.ptr.rhs.cast646 = ptrtoint i8* %line to i64
  %old = alloca [512 x i8], align 16
  %0 = getelementptr inbounds [512 x i8], [512 x i8]* %old, i64 0, i64 0
  switch i64 %fid, label %if.then [
    i64 2, label %if.end
    i64 0, label %if.end
  ]

if.then:
  br label %cleanup

if.end:
  switch i64 undef, label %if.end25 [
    i64 0, label %if.then4
    i64 1, label %if.end25
  ]

if.then4:
  br i1 undef, label %SyTime.exit, label %if.then.i

if.then.i:
  unreachable

SyTime.exit:
  br i1 undef, label %SyTime.exit2681, label %if.then.i2673

if.then.i2673:
  unreachable

SyTime.exit2681:
  br label %cleanup

land.lhs.true14:
  unreachable

if.end25:
  br i1 undef, label %SyTime.exit2720, label %if.then.i2712

if.then.i2712:
  unreachable

SyTime.exit2720:
  %add.ptr = getelementptr [512 x i8], [512 x i8]* %old, i64 0, i64 512
  %cmp293427 = icmp ult i8* %0, %add.ptr
  br i1 %cmp293427, label %for.body.lr.ph, label %while.body.preheader

for.body.lr.ph:
  call void @llvm.memset.p0i8.i64(i8* align 16 undef, i8 32, i64 512, i1 false)
  br label %while.body.preheader

while.body.preheader:
  %add.ptr1603 = getelementptr [512 x i8], [512 x i8]* null, i64 0, i64 512
  %echo.i3101 = getelementptr [16 x %struct.TMP.1], [16 x %struct.TMP.1]* @syBuf, i64 0, i64 %fid, i32 1
  %1 = xor i64 %sub.ptr.rhs.cast646, -1
  br label %do.body

do.body:
  %ch2.0 = phi i32 [ 0, %while.body.preheader ], [ %ch.12.ch2.12, %do.body ]
  %rep.0 = phi i32 [ 1, %while.body.preheader ], [ %rep.6, %do.body ]
  store i32 0, i32* @syCTRO, align 4, !tbaa !1
  %ch.0.ch2.0 = select i1 undef, i32 14, i32 %ch2.0
  %ch2.2 = select i1 undef, i32 0, i32 %ch.0.ch2.0
  %ch.2.ch2.2 = select i1 undef, i32 0, i32 %ch2.2
  %ch2.4 = select i1 undef, i32 278, i32 %ch.2.ch2.2
  %ch2.5 = select i1 undef, i32 0, i32 %ch2.4
  %rep.2 = select i1 undef, i32 undef, i32 %rep.0
  %ch.5.ch2.5 = select i1 undef, i32 undef, i32 %ch2.5
  %ch2.7 = select i1 undef, i32 0, i32 %ch.5.ch2.5
  %rep.3 = select i1 undef, i32 undef, i32 %rep.2
  %ch.7.ch2.7 = select i1 false, i32 0, i32 %ch2.7
  %mul98.rep.3 = select i1 false, i32 0, i32 %rep.3
  %ch2.9 = select i1 undef, i32 undef, i32 %ch.7.ch2.7
  %rep.5 = select i1 undef, i32 undef, i32 %mul98.rep.3
  %ch2.10 = select i1 false, i32 undef, i32 %ch2.9
  %rep.6 = select i1 false, i32 undef, i32 %rep.5
  %isdigittmp = add i32 %ch2.10, -48
  %isdigit = icmp ult i32 %isdigittmp, 10
  %cmp119 = icmp eq i32 undef, 22
  %or.cond1875 = and i1 %isdigit, %cmp119
  %ch.10.ch2.10 = select i1 %or.cond1875, i32 undef, i32 %ch2.10
  %.ch.10 = select i1 %or.cond1875, i32 0, i32 undef
  %ch2.12 = select i1 undef, i32 %.ch.10, i32 %ch.10.ch2.10
  %ch.12 = select i1 undef, i32 0, i32 %.ch.10
  %ch.12.ch2.12 = select i1 false, i32 %ch.12, i32 %ch2.12
  %.ch.12 = select i1 false, i32 0, i32 %ch.12
  %cmp147 = icmp eq i32 %.ch.12, 0
  br i1 %cmp147, label %do.body, label %do.end

do.end:
  %cmp164 = icmp eq i32 %ch.12.ch2.12, 21
  %mul167 = shl i32 %rep.6, 2
  %rep.8 = select i1 %cmp164, i32 %mul167, i32 %rep.6
  %..ch.19 = select i1 false, i32 2, i32 0
  br i1 undef, label %while.body200, label %while.end1465

while.body200:
  %dec3386.in = phi i32 [ %dec3386, %while.cond197.backedge ], [ %rep.8, %do.end ]
  %oldc.13384 = phi i32 [ %oldc.1.be, %while.cond197.backedge ], [ 0, %do.end ]
  %ch.213379 = phi i32 [ %last.1.be, %while.cond197.backedge ], [ %..ch.19, %do.end ]
  %last.13371 = phi i32 [ %last.1.be, %while.cond197.backedge ], [ 0, %do.end ]
  %dec3386 = add i32 %dec3386.in, -1
  switch i32 %ch.213379, label %sw.default [
    i32 1, label %while.cond201.preheader
    i32 322, label %sw.bb206
    i32 354, label %sw.bb206
    i32 2, label %sw.bb243
    i32 364, label %sw.bb1077
    i32 326, label %sw.bb256
    i32 358, label %sw.bb256
    i32 341, label %sw.bb979
    i32 323, label %while.cond1037.preheader
    i32 373, label %sw.bb979
    i32 4, label %if.then1477
    i32 332, label %sw.bb1077
    i32 11, label %for.cond357
    i32 355, label %while.cond1037.preheader
    i32 324, label %sw.bb474
    i32 356, label %sw.bb474
    i32 20, label %sw.bb566
    i32 -1, label %while.cond197.backedge
    i32 268, label %sw.bb1134
    i32 16, label %while.cond635.preheader
    i32 18, label %sw.bb956
    i32 316, label %while.cond864
  ]

while.cond1037.preheader:
  %cmp10393273 = icmp eq i8 undef, 0
  br i1 %cmp10393273, label %if.end1070, label %land.rhs1041

while.cond635.preheader:
  br i1 undef, label %for.body643.us, label %while.cond661

for.body643.us:
  br label %for.body643.us

while.cond201.preheader:
  %umax = select i1 false, i64 undef, i64 %1
  %2 = xor i64 %umax, -1
  %3 = inttoptr i64 %2 to i8*
  br label %while.cond197.backedge

sw.bb206:
  br label %while.cond197.backedge

sw.bb243:
  br label %while.cond197.backedge

sw.bb256:
  br label %while.cond197.backedge

while.cond197.backedge:
  %last.1.be = phi i32 [ %ch.213379, %sw.default ], [ -1, %while.body200 ], [ %ch.213379, %sw.bb1077 ], [ %ch.213379, %sw.bb979 ], [ 18, %sw.bb956 ], [ 20, %sw.bb566 ], [ %ch.213379, %for.end552 ], [ %ch.213379, %sw.bb256 ], [ 2, %sw.bb243 ], [ 1, %while.cond201.preheader ], [ 268, %for.cond1145.preheader ], [ %ch.213379, %sw.bb206 ]
  %oldc.1.be = phi i32 [ %oldc.13384, %sw.default ], [ %oldc.13384, %while.body200 ], [ %oldc.13384, %sw.bb1077 ], [ %oldc.13384, %sw.bb979 ], [ %oldc.13384, %sw.bb956 ], [ %oldc.13384, %sw.bb566 ], [ %oldc.13384, %for.end552 ], [ %oldc.13384, %sw.bb256 ], [ %oldc.13384, %sw.bb243 ], [ %oldc.13384, %while.cond201.preheader ], [ 0, %for.cond1145.preheader ], [ %oldc.13384, %sw.bb206 ]
  %cmp198 = icmp sgt i32 %dec3386, 0
  br i1 %cmp198, label %while.body200, label %while.end1465

for.cond357:
  br label %for.cond357

sw.bb474:
  ; spill is hoisted here. Although loop depth1 is even hotter than loop depth2, sw.bb474 is still cold.
  %cmp476 = icmp eq i8 undef, 0
  br i1 %cmp476, label %if.end517, label %do.body479.preheader

do.body479.preheader:
  %cmp4833314 = icmp eq i8 undef, 0
  br i1 %cmp4833314, label %if.end517, label %land.rhs485

land.rhs485:
  %incdec.ptr4803316 = phi i8* [ %incdec.ptr480, %do.body479.backedge.land.rhs485_crit_edge ], [ undef, %do.body479.preheader ]
  %isascii.i.i27763151 = icmp sgt i8 undef, -1
  br i1 %isascii.i.i27763151, label %cond.true.i.i2780, label %cond.false.i.i2782

cond.true.i.i2780:
  br i1 undef, label %land.lhs.true490, label %lor.rhs500

cond.false.i.i2782:
  unreachable

land.lhs.true490:
  br i1 false, label %lor.rhs500, label %do.body479.backedge

lor.rhs500:
  ; Make sure spill is hoisted to a cold preheader in outside loop.
  %call3.i.i2792 = call i32 @__maskrune(i32 undef, i64 256)
  br i1 undef, label %land.lhs.true504, label %do.body479.backedge

land.lhs.true504:
  br i1 undef, label %do.body479.backedge, label %if.end517

do.body479.backedge:
  %incdec.ptr480 = getelementptr i8, i8* %incdec.ptr4803316, i64 1
  %cmp483 = icmp eq i8 undef, 0
  br i1 %cmp483, label %if.end517, label %do.body479.backedge.land.rhs485_crit_edge

do.body479.backedge.land.rhs485_crit_edge:
  br label %land.rhs485

if.end517:
  %q.4 = phi i8* [ undef, %sw.bb474 ], [ undef, %do.body479.preheader ], [ %incdec.ptr480, %do.body479.backedge ], [ %incdec.ptr4803316, %land.lhs.true504 ]
  switch i32 %last.13371, label %if.then532 [
    i32 383, label %for.cond534
    i32 356, label %for.cond534
    i32 324, label %for.cond534
    i32 24, label %for.cond534
    i32 11, label %for.cond534
  ]

if.then532:
  store i8 0, i8* getelementptr inbounds ([512 x i8], [512 x i8]* @SyFgets.yank, i64 0, i64 0), align 16, !tbaa !5
  br label %for.cond534

for.cond534:
  %cmp536 = icmp eq i8 undef, 0
  br i1 %cmp536, label %for.cond542.preheader, label %for.cond534

for.cond542.preheader:
  br i1 undef, label %for.body545, label %for.end552

for.body545:
  br i1 undef, label %for.end552, label %for.body545

for.end552:
  %s.2.lcssa = phi i8* [ undef, %for.cond542.preheader ], [ %q.4, %for.body545 ]
  %sub.ptr.lhs.cast553 = ptrtoint i8* %s.2.lcssa to i64
  %sub.ptr.sub555 = sub i64 %sub.ptr.lhs.cast553, 0
  %arrayidx556 = getelementptr i8, i8* null, i64 %sub.ptr.sub555
  store i8 0, i8* %arrayidx556, align 1, !tbaa !5
  br label %while.cond197.backedge

sw.bb566:
  br label %while.cond197.backedge

while.cond661:
  br label %while.cond661

while.cond864:
  br label %while.cond864

sw.bb956:
  br i1 undef, label %if.then959, label %while.cond197.backedge

if.then959:
  br label %while.cond962

while.cond962:
  br label %while.cond962

sw.bb979:
  br label %while.cond197.backedge

land.rhs1041:
  unreachable

if.end1070:
  br label %sw.bb1077

sw.bb1077:
  br label %while.cond197.backedge

sw.bb1134:
  br i1 false, label %for.body1139, label %for.cond1145.preheader

for.cond1145.preheader:
  br i1 %cmp293427, label %for.body1150.lr.ph, label %while.cond197.backedge

for.body1150.lr.ph:
  unreachable

for.body1139:
  unreachable

sw.default:
  br label %while.cond197.backedge

while.end1465:
  %oldc.1.lcssa = phi i32 [ 0, %do.end ], [ %oldc.1.be, %while.cond197.backedge ]
  %ch.21.lcssa = phi i32 [ %..ch.19, %do.end ], [ %last.1.be, %while.cond197.backedge ]
  switch i32 %ch.21.lcssa, label %for.cond1480.preheader [
    i32 -1, label %if.then1477
    i32 15, label %if.then1477
    i32 13, label %if.then1477
    i32 10, label %if.then1477
  ]

for.cond1480.preheader:
  br i1 undef, label %for.body1606.lr.ph, label %for.end1609

if.then1477:
  %p.1.lcssa3539 = phi i8* [ null, %while.end1465 ], [ null, %while.end1465 ], [ null, %while.end1465 ], [ null, %while.end1465 ], [ %line, %while.body200 ]
  %call1.i3057 = call i64 @"\01_write"(i32 undef, i8* undef, i64 1)
  %sub.ptr.lhs.cast1717 = ptrtoint i8* %p.1.lcssa3539 to i64
  %sub.ptr.sub1719 = sub i64 %sub.ptr.lhs.cast1717, %sub.ptr.rhs.cast646
  %idx.neg1727 = sub i64 0, %sub.ptr.sub1719
  br label %for.body1723

for.body1606.lr.ph:
  br label %for.end1609

for.end1609:
  br i1 undef, label %for.cond1659.preheader, label %land.lhs.true1614

land.lhs.true1614:
  br label %for.cond1659.preheader

for.cond1659.preheader:
  %cmp16623414 = icmp ult i8* undef, %add.ptr1603
  br i1 %cmp16623414, label %for.body1664.lr.ph, label %while.body1703.lr.ph

for.body1664.lr.ph:
  %cmp16773405 = icmp slt i64 undef, undef
  br i1 %cmp16773405, label %while.body1679, label %while.cond1683.preheader

while.body1703.lr.ph:
  unreachable

while.cond1683.preheader:
  br i1 undef, label %while.body1691, label %while.end1693

while.body1679:
  %oldc.43406 = phi i32 [ %inc, %syEchoch.exit3070 ], [ %oldc.1.lcssa, %for.body1664.lr.ph ]
  %4 = load %struct.TMP.2*, %struct.TMP.2** %echo.i3101, align 8, !tbaa !6
  %call.i3062 = call i32 @fileno(%struct.TMP.2* %4)
  br i1 undef, label %if.then.i3069, label %syEchoch.exit3070

if.then.i3069:
  br label %syEchoch.exit3070

syEchoch.exit3070:
  %inc = add i32 %oldc.43406, 1
  %conv1672 = sext i32 %inc to i64
  %cmp1677 = icmp slt i64 %conv1672, undef
  br i1 %cmp1677, label %while.body1679, label %while.cond1683.preheader

while.body1691:
  unreachable

while.end1693:
  unreachable

for.body1723:
  %q.303203 = phi i8* [ getelementptr inbounds ([8192 x i8], [8192 x i8]* @syHistory, i64 0, i64 8189), %if.then1477 ], [ %incdec.ptr1730, %for.body1723 ]
  %add.ptr1728 = getelementptr i8, i8* %q.303203, i64 %idx.neg1727
  %5 = load i8, i8* %add.ptr1728, align 1, !tbaa !5
  %incdec.ptr1730 = getelementptr i8, i8* %q.303203, i64 -1
  br label %for.body1723

cleanup:
  ret i8* undef
}

declare i32 @fileno(%struct.TMP.2* nocapture)
declare i64 @"\01_write"(i32, i8*, i64)
declare i32 @__maskrune(i32, i64)
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 (trunk 204257)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!3, !3, i64 0}
!6 = !{!7, !8, i64 8}
!7 = !{!"", !8, i64 0, !8, i64 8, !3, i64 16}
!8 = !{!"any pointer", !3, i64 0}
