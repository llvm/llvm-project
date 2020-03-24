; RUN: opt < %s -passes=loop-spawning -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN8sequence4packIllN5utils9identityFIlEEEE4_seqIT_EPS5_PbT0_S9_T1_ = comdat any

; Function Attrs: uwtable
define linkonce_odr { i64*, i64 } @_ZN8sequence4packIllN5utils9identityFIlEEEE4_seqIT_EPS5_PbT0_S9_T1_(i64* %Out, i8* %Fl, i64 %s, i64 %e) local_unnamed_addr #3 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg53 = tail call token @llvm.syncregion.start()
  %sub = sub nsw i64 %e, %s
  %sub1 = add nsw i64 %sub, -1
  %div = sdiv i64 %sub1, 2048
  %add = add nsw i64 %div, 1
  %cmp = icmp slt i64 %sub, 2049
  br i1 %cmp, label %if.then, label %pfor.detach.lr.ph

if.then:                                          ; preds = %entry
  %cmp.i = icmp eq i64* %Out, null
  br i1 %cmp.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %if.then
  %add.ptr.i = getelementptr inbounds i8, i8* %Fl, i64 %s
  %cmp.i.i = icmp sgt i64 %sub, 127
  %and.i.i = and i64 %sub, 511
  %cmp1.i.i = icmp eq i64 %and.i.i, 0
  %or.cond.i.i = and i1 %cmp.i.i, %cmp1.i.i
  br i1 %or.cond.i.i, label %land.lhs.true2.i.i, label %if.else.i.i

land.lhs.true2.i.i:                               ; preds = %if.then.i
  %0 = ptrtoint i8* %add.ptr.i to i64
  %and3.i.i = and i64 %0, 3
  %cmp4.i.i = icmp eq i64 %and3.i.i, 0
  br i1 %cmp4.i.i, label %if.then.i.i, label %for.body29.lr.ph.i.i

if.then.i.i:                                      ; preds = %land.lhs.true2.i.i
  %shr75.i.i = lshr i64 %sub, 9
  %cmp562.i.i = icmp sgt i64 %sub, 511
  br i1 %cmp562.i.i, label %for.body.lr.ph.i.i, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i

for.body.lr.ph.i.i:                               ; preds = %if.then.i.i
  %1 = bitcast i8* %add.ptr.i to i32*
  br label %vector.ph452

vector.ph452:                                     ; preds = %for.body.lr.ph.i.i, %vector.ph452
  %indvars.iv71.i.i = phi i64 [ 0, %for.body.lr.ph.i.i ], [ %indvars.iv.next72.i.i, %vector.ph452 ]
  %IFl.064.i.i = phi i32* [ %1, %for.body.lr.ph.i.i ], [ %add.ptr.i.i, %vector.ph452 ]
  %r.063.i.i = phi i64 [ 0, %for.body.lr.ph.i.i ], [ %add21.i.i, %vector.ph452 ]
  %2 = bitcast i32* %IFl.064.i.i to <8 x i32>*
  %wide.load467 = load <8 x i32>, <8 x i32>* %2, align 4, !tbaa !55
  %3 = getelementptr i32, i32* %IFl.064.i.i, i64 8
  %4 = bitcast i32* %3 to <8 x i32>*
  %wide.load468 = load <8 x i32>, <8 x i32>* %4, align 4, !tbaa !55
  %5 = getelementptr i32, i32* %IFl.064.i.i, i64 16
  %6 = bitcast i32* %5 to <8 x i32>*
  %wide.load469 = load <8 x i32>, <8 x i32>* %6, align 4, !tbaa !55
  %7 = getelementptr i32, i32* %IFl.064.i.i, i64 24
  %8 = bitcast i32* %7 to <8 x i32>*
  %wide.load470 = load <8 x i32>, <8 x i32>* %8, align 4, !tbaa !55
  %9 = getelementptr inbounds i32, i32* %IFl.064.i.i, i64 32
  %10 = bitcast i32* %9 to <8 x i32>*
  %wide.load467.1 = load <8 x i32>, <8 x i32>* %10, align 4, !tbaa !55
  %11 = getelementptr i32, i32* %IFl.064.i.i, i64 40
  %12 = bitcast i32* %11 to <8 x i32>*
  %wide.load468.1 = load <8 x i32>, <8 x i32>* %12, align 4, !tbaa !55
  %13 = getelementptr i32, i32* %IFl.064.i.i, i64 48
  %14 = bitcast i32* %13 to <8 x i32>*
  %wide.load469.1 = load <8 x i32>, <8 x i32>* %14, align 4, !tbaa !55
  %15 = getelementptr i32, i32* %IFl.064.i.i, i64 56
  %16 = bitcast i32* %15 to <8 x i32>*
  %wide.load470.1 = load <8 x i32>, <8 x i32>* %16, align 4, !tbaa !55
  %17 = add nsw <8 x i32> %wide.load467.1, %wide.load467
  %18 = add nsw <8 x i32> %wide.load468.1, %wide.load468
  %19 = add nsw <8 x i32> %wide.load469.1, %wide.load469
  %20 = add nsw <8 x i32> %wide.load470.1, %wide.load470
  %21 = getelementptr inbounds i32, i32* %IFl.064.i.i, i64 64
  %22 = bitcast i32* %21 to <8 x i32>*
  %wide.load467.2 = load <8 x i32>, <8 x i32>* %22, align 4, !tbaa !55
  %23 = getelementptr i32, i32* %IFl.064.i.i, i64 72
  %24 = bitcast i32* %23 to <8 x i32>*
  %wide.load468.2 = load <8 x i32>, <8 x i32>* %24, align 4, !tbaa !55
  %25 = getelementptr i32, i32* %IFl.064.i.i, i64 80
  %26 = bitcast i32* %25 to <8 x i32>*
  %wide.load469.2 = load <8 x i32>, <8 x i32>* %26, align 4, !tbaa !55
  %27 = getelementptr i32, i32* %IFl.064.i.i, i64 88
  %28 = bitcast i32* %27 to <8 x i32>*
  %wide.load470.2 = load <8 x i32>, <8 x i32>* %28, align 4, !tbaa !55
  %29 = add nsw <8 x i32> %wide.load467.2, %17
  %30 = add nsw <8 x i32> %wide.load468.2, %18
  %31 = add nsw <8 x i32> %wide.load469.2, %19
  %32 = add nsw <8 x i32> %wide.load470.2, %20
  %33 = getelementptr inbounds i32, i32* %IFl.064.i.i, i64 96
  %34 = bitcast i32* %33 to <8 x i32>*
  %wide.load467.3 = load <8 x i32>, <8 x i32>* %34, align 4, !tbaa !55
  %35 = getelementptr i32, i32* %IFl.064.i.i, i64 104
  %36 = bitcast i32* %35 to <8 x i32>*
  %wide.load468.3 = load <8 x i32>, <8 x i32>* %36, align 4, !tbaa !55
  %37 = getelementptr i32, i32* %IFl.064.i.i, i64 112
  %38 = bitcast i32* %37 to <8 x i32>*
  %wide.load469.3 = load <8 x i32>, <8 x i32>* %38, align 4, !tbaa !55
  %39 = getelementptr i32, i32* %IFl.064.i.i, i64 120
  %40 = bitcast i32* %39 to <8 x i32>*
  %wide.load470.3 = load <8 x i32>, <8 x i32>* %40, align 4, !tbaa !55
  %41 = add nsw <8 x i32> %wide.load467.3, %29
  %42 = add nsw <8 x i32> %wide.load468.3, %30
  %43 = add nsw <8 x i32> %wide.load469.3, %31
  %44 = add nsw <8 x i32> %wide.load470.3, %32
  %bin.rdx471 = add <8 x i32> %42, %41
  %bin.rdx472 = add <8 x i32> %43, %bin.rdx471
  %bin.rdx473 = add <8 x i32> %44, %bin.rdx472
  %rdx.shuf474 = shufflevector <8 x i32> %bin.rdx473, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx475 = add <8 x i32> %bin.rdx473, %rdx.shuf474
  %rdx.shuf476 = shufflevector <8 x i32> %bin.rdx475, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx477 = add <8 x i32> %bin.rdx475, %rdx.shuf476
  %rdx.shuf478 = shufflevector <8 x i32> %bin.rdx477, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx479 = add <8 x i32> %bin.rdx477, %rdx.shuf478
  %45 = extractelement <8 x i32> %bin.rdx479, i32 0
  %and10.i.i = and i32 %45, 255
  %46 = lshr i32 %45, 8
  %and12.i.i = and i32 %46, 255
  %47 = lshr i32 %45, 16
  %and15.i.i = and i32 %47, 255
  %48 = lshr i32 %45, 24
  %add13.i.i = add nuw nsw i32 %48, %and10.i.i
  %add16.i.i = add nuw nsw i32 %add13.i.i, %and12.i.i
  %add19.i.i = add nuw nsw i32 %add16.i.i, %and15.i.i
  %49 = zext i32 %add19.i.i to i64
  %add21.i.i = add nuw nsw i64 %r.063.i.i, %49
  %add.ptr.i.i = getelementptr inbounds i32, i32* %IFl.064.i.i, i64 128
  %indvars.iv.next72.i.i = add nuw nsw i64 %indvars.iv71.i.i, 1
  %cmp5.i.i = icmp ugt i64 %shr75.i.i, %indvars.iv.next72.i.i
  br i1 %cmp5.i.i, label %vector.ph452, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i

if.else.i.i:                                      ; preds = %if.then.i
  %cmp2766.i.i = icmp sgt i64 %sub, 0
  br i1 %cmp2766.i.i, label %for.body29.lr.ph.i.i, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i

for.body29.lr.ph.i.i:                             ; preds = %if.else.i.i, %land.lhs.true2.i.i
  %min.iters.check419 = icmp ult i64 %sub, 16
  br i1 %min.iters.check419, label %for.body29.i.i.preheader, label %vector.ph420

vector.ph420:                                     ; preds = %for.body29.lr.ph.i.i
  %n.vec422 = and i64 %sub, -16
  %50 = add i64 %n.vec422, -16
  %51 = lshr exact i64 %50, 4
  %52 = add nuw nsw i64 %51, 1
  %xtraiter508 = and i64 %52, 3
  %53 = icmp ult i64 %50, 48
  br i1 %53, label %middle.block417.unr-lcssa, label %vector.ph420.new

vector.ph420.new:                                 ; preds = %vector.ph420
  %unroll_iter515 = sub nsw i64 %52, %xtraiter508
  br label %vector.body416

vector.body416:                                   ; preds = %vector.body416, %vector.ph420.new
  %index423 = phi i64 [ 0, %vector.ph420.new ], [ %index.next424.3, %vector.body416 ]
  %vec.phi433 = phi <4 x i64> [ zeroinitializer, %vector.ph420.new ], [ %114, %vector.body416 ]
  %vec.phi434 = phi <4 x i64> [ zeroinitializer, %vector.ph420.new ], [ %115, %vector.body416 ]
  %vec.phi435 = phi <4 x i64> [ zeroinitializer, %vector.ph420.new ], [ %116, %vector.body416 ]
  %vec.phi436 = phi <4 x i64> [ zeroinitializer, %vector.ph420.new ], [ %117, %vector.body416 ]
  %niter516 = phi i64 [ %unroll_iter515, %vector.ph420.new ], [ %niter516.nsub.3, %vector.body416 ]
  %54 = getelementptr inbounds i8, i8* %add.ptr.i, i64 %index423
  %55 = bitcast i8* %54 to <4 x i8>*
  %wide.load437 = load <4 x i8>, <4 x i8>* %55, align 1, !tbaa !7
  %56 = getelementptr i8, i8* %54, i64 4
  %57 = bitcast i8* %56 to <4 x i8>*
  %wide.load438 = load <4 x i8>, <4 x i8>* %57, align 1, !tbaa !7
  %58 = getelementptr i8, i8* %54, i64 8
  %59 = bitcast i8* %58 to <4 x i8>*
  %wide.load439 = load <4 x i8>, <4 x i8>* %59, align 1, !tbaa !7
  %60 = getelementptr i8, i8* %54, i64 12
  %61 = bitcast i8* %60 to <4 x i8>*
  %wide.load440 = load <4 x i8>, <4 x i8>* %61, align 1, !tbaa !7
  %62 = zext <4 x i8> %wide.load437 to <4 x i64>
  %63 = zext <4 x i8> %wide.load438 to <4 x i64>
  %64 = zext <4 x i8> %wide.load439 to <4 x i64>
  %65 = zext <4 x i8> %wide.load440 to <4 x i64>
  %66 = add nuw nsw <4 x i64> %vec.phi433, %62
  %67 = add nuw nsw <4 x i64> %vec.phi434, %63
  %68 = add nuw nsw <4 x i64> %vec.phi435, %64
  %69 = add nuw nsw <4 x i64> %vec.phi436, %65
  %index.next424 = or i64 %index423, 16
  %70 = getelementptr inbounds i8, i8* %add.ptr.i, i64 %index.next424
  %71 = bitcast i8* %70 to <4 x i8>*
  %wide.load437.1 = load <4 x i8>, <4 x i8>* %71, align 1, !tbaa !7
  %72 = getelementptr i8, i8* %70, i64 4
  %73 = bitcast i8* %72 to <4 x i8>*
  %wide.load438.1 = load <4 x i8>, <4 x i8>* %73, align 1, !tbaa !7
  %74 = getelementptr i8, i8* %70, i64 8
  %75 = bitcast i8* %74 to <4 x i8>*
  %wide.load439.1 = load <4 x i8>, <4 x i8>* %75, align 1, !tbaa !7
  %76 = getelementptr i8, i8* %70, i64 12
  %77 = bitcast i8* %76 to <4 x i8>*
  %wide.load440.1 = load <4 x i8>, <4 x i8>* %77, align 1, !tbaa !7
  %78 = zext <4 x i8> %wide.load437.1 to <4 x i64>
  %79 = zext <4 x i8> %wide.load438.1 to <4 x i64>
  %80 = zext <4 x i8> %wide.load439.1 to <4 x i64>
  %81 = zext <4 x i8> %wide.load440.1 to <4 x i64>
  %82 = add nuw nsw <4 x i64> %66, %78
  %83 = add nuw nsw <4 x i64> %67, %79
  %84 = add nuw nsw <4 x i64> %68, %80
  %85 = add nuw nsw <4 x i64> %69, %81
  %index.next424.1 = or i64 %index423, 32
  %86 = getelementptr inbounds i8, i8* %add.ptr.i, i64 %index.next424.1
  %87 = bitcast i8* %86 to <4 x i8>*
  %wide.load437.2 = load <4 x i8>, <4 x i8>* %87, align 1, !tbaa !7
  %88 = getelementptr i8, i8* %86, i64 4
  %89 = bitcast i8* %88 to <4 x i8>*
  %wide.load438.2 = load <4 x i8>, <4 x i8>* %89, align 1, !tbaa !7
  %90 = getelementptr i8, i8* %86, i64 8
  %91 = bitcast i8* %90 to <4 x i8>*
  %wide.load439.2 = load <4 x i8>, <4 x i8>* %91, align 1, !tbaa !7
  %92 = getelementptr i8, i8* %86, i64 12
  %93 = bitcast i8* %92 to <4 x i8>*
  %wide.load440.2 = load <4 x i8>, <4 x i8>* %93, align 1, !tbaa !7
  %94 = zext <4 x i8> %wide.load437.2 to <4 x i64>
  %95 = zext <4 x i8> %wide.load438.2 to <4 x i64>
  %96 = zext <4 x i8> %wide.load439.2 to <4 x i64>
  %97 = zext <4 x i8> %wide.load440.2 to <4 x i64>
  %98 = add nuw nsw <4 x i64> %82, %94
  %99 = add nuw nsw <4 x i64> %83, %95
  %100 = add nuw nsw <4 x i64> %84, %96
  %101 = add nuw nsw <4 x i64> %85, %97
  %index.next424.2 = or i64 %index423, 48
  %102 = getelementptr inbounds i8, i8* %add.ptr.i, i64 %index.next424.2
  %103 = bitcast i8* %102 to <4 x i8>*
  %wide.load437.3 = load <4 x i8>, <4 x i8>* %103, align 1, !tbaa !7
  %104 = getelementptr i8, i8* %102, i64 4
  %105 = bitcast i8* %104 to <4 x i8>*
  %wide.load438.3 = load <4 x i8>, <4 x i8>* %105, align 1, !tbaa !7
  %106 = getelementptr i8, i8* %102, i64 8
  %107 = bitcast i8* %106 to <4 x i8>*
  %wide.load439.3 = load <4 x i8>, <4 x i8>* %107, align 1, !tbaa !7
  %108 = getelementptr i8, i8* %102, i64 12
  %109 = bitcast i8* %108 to <4 x i8>*
  %wide.load440.3 = load <4 x i8>, <4 x i8>* %109, align 1, !tbaa !7
  %110 = zext <4 x i8> %wide.load437.3 to <4 x i64>
  %111 = zext <4 x i8> %wide.load438.3 to <4 x i64>
  %112 = zext <4 x i8> %wide.load439.3 to <4 x i64>
  %113 = zext <4 x i8> %wide.load440.3 to <4 x i64>
  %114 = add nuw nsw <4 x i64> %98, %110
  %115 = add nuw nsw <4 x i64> %99, %111
  %116 = add nuw nsw <4 x i64> %100, %112
  %117 = add nuw nsw <4 x i64> %101, %113
  %index.next424.3 = add i64 %index423, 64
  %niter516.nsub.3 = add i64 %niter516, -4
  %niter516.ncmp.3 = icmp eq i64 %niter516.nsub.3, 0
  br i1 %niter516.ncmp.3, label %middle.block417.unr-lcssa, label %vector.body416, !llvm.loop !96

middle.block417.unr-lcssa:                        ; preds = %vector.body416, %vector.ph420
  %.lcssa490.ph = phi <4 x i64> [ undef, %vector.ph420 ], [ %114, %vector.body416 ]
  %.lcssa489.ph = phi <4 x i64> [ undef, %vector.ph420 ], [ %115, %vector.body416 ]
  %.lcssa488.ph = phi <4 x i64> [ undef, %vector.ph420 ], [ %116, %vector.body416 ]
  %.lcssa487.ph = phi <4 x i64> [ undef, %vector.ph420 ], [ %117, %vector.body416 ]
  %index423.unr = phi i64 [ 0, %vector.ph420 ], [ %index.next424.3, %vector.body416 ]
  %vec.phi433.unr = phi <4 x i64> [ zeroinitializer, %vector.ph420 ], [ %114, %vector.body416 ]
  %vec.phi434.unr = phi <4 x i64> [ zeroinitializer, %vector.ph420 ], [ %115, %vector.body416 ]
  %vec.phi435.unr = phi <4 x i64> [ zeroinitializer, %vector.ph420 ], [ %116, %vector.body416 ]
  %vec.phi436.unr = phi <4 x i64> [ zeroinitializer, %vector.ph420 ], [ %117, %vector.body416 ]
  %lcmp.mod510 = icmp eq i64 %xtraiter508, 0
  br i1 %lcmp.mod510, label %middle.block417, label %vector.body416.epil.preheader

vector.body416.epil.preheader:                    ; preds = %middle.block417.unr-lcssa
  br label %vector.body416.epil

vector.body416.epil:                              ; preds = %vector.body416.epil, %vector.body416.epil.preheader
  %index423.epil = phi i64 [ %index423.unr, %vector.body416.epil.preheader ], [ %index.next424.epil, %vector.body416.epil ]
  %vec.phi433.epil = phi <4 x i64> [ %vec.phi433.unr, %vector.body416.epil.preheader ], [ %130, %vector.body416.epil ]
  %vec.phi434.epil = phi <4 x i64> [ %vec.phi434.unr, %vector.body416.epil.preheader ], [ %131, %vector.body416.epil ]
  %vec.phi435.epil = phi <4 x i64> [ %vec.phi435.unr, %vector.body416.epil.preheader ], [ %132, %vector.body416.epil ]
  %vec.phi436.epil = phi <4 x i64> [ %vec.phi436.unr, %vector.body416.epil.preheader ], [ %133, %vector.body416.epil ]
  %epil.iter509 = phi i64 [ %xtraiter508, %vector.body416.epil.preheader ], [ %epil.iter509.sub, %vector.body416.epil ]
  %118 = getelementptr inbounds i8, i8* %add.ptr.i, i64 %index423.epil
  %119 = bitcast i8* %118 to <4 x i8>*
  %wide.load437.epil = load <4 x i8>, <4 x i8>* %119, align 1, !tbaa !7
  %120 = getelementptr i8, i8* %118, i64 4
  %121 = bitcast i8* %120 to <4 x i8>*
  %wide.load438.epil = load <4 x i8>, <4 x i8>* %121, align 1, !tbaa !7
  %122 = getelementptr i8, i8* %118, i64 8
  %123 = bitcast i8* %122 to <4 x i8>*
  %wide.load439.epil = load <4 x i8>, <4 x i8>* %123, align 1, !tbaa !7
  %124 = getelementptr i8, i8* %118, i64 12
  %125 = bitcast i8* %124 to <4 x i8>*
  %wide.load440.epil = load <4 x i8>, <4 x i8>* %125, align 1, !tbaa !7
  %126 = zext <4 x i8> %wide.load437.epil to <4 x i64>
  %127 = zext <4 x i8> %wide.load438.epil to <4 x i64>
  %128 = zext <4 x i8> %wide.load439.epil to <4 x i64>
  %129 = zext <4 x i8> %wide.load440.epil to <4 x i64>
  %130 = add nuw nsw <4 x i64> %vec.phi433.epil, %126
  %131 = add nuw nsw <4 x i64> %vec.phi434.epil, %127
  %132 = add nuw nsw <4 x i64> %vec.phi435.epil, %128
  %133 = add nuw nsw <4 x i64> %vec.phi436.epil, %129
  %index.next424.epil = add i64 %index423.epil, 16
  %epil.iter509.sub = add i64 %epil.iter509, -1
  %epil.iter509.cmp = icmp eq i64 %epil.iter509.sub, 0
  br i1 %epil.iter509.cmp, label %middle.block417, label %vector.body416.epil, !llvm.loop !97

middle.block417:                                  ; preds = %vector.body416.epil, %middle.block417.unr-lcssa
  %.lcssa490 = phi <4 x i64> [ %.lcssa490.ph, %middle.block417.unr-lcssa ], [ %130, %vector.body416.epil ]
  %.lcssa489 = phi <4 x i64> [ %.lcssa489.ph, %middle.block417.unr-lcssa ], [ %131, %vector.body416.epil ]
  %.lcssa488 = phi <4 x i64> [ %.lcssa488.ph, %middle.block417.unr-lcssa ], [ %132, %vector.body416.epil ]
  %.lcssa487 = phi <4 x i64> [ %.lcssa487.ph, %middle.block417.unr-lcssa ], [ %133, %vector.body416.epil ]
  %bin.rdx441 = add nuw <4 x i64> %.lcssa489, %.lcssa490
  %bin.rdx442 = add <4 x i64> %.lcssa488, %bin.rdx441
  %bin.rdx443 = add <4 x i64> %.lcssa487, %bin.rdx442
  %rdx.shuf444 = shufflevector <4 x i64> %bin.rdx443, <4 x i64> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx445 = add <4 x i64> %bin.rdx443, %rdx.shuf444
  %rdx.shuf446 = shufflevector <4 x i64> %bin.rdx445, <4 x i64> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx447 = add <4 x i64> %bin.rdx445, %rdx.shuf446
  %134 = extractelement <4 x i64> %bin.rdx447, i32 0
  %cmp.n426 = icmp eq i64 %sub, %n.vec422
  br i1 %cmp.n426, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i, label %for.body29.i.i.preheader

for.body29.i.i.preheader:                         ; preds = %middle.block417, %for.body29.lr.ph.i.i
  %j25.068.i.i.ph = phi i64 [ 0, %for.body29.lr.ph.i.i ], [ %n.vec422, %middle.block417 ]
  %r.167.i.i.ph = phi i64 [ 0, %for.body29.lr.ph.i.i ], [ %134, %middle.block417 ]
  br label %for.body29.i.i

for.body29.i.i:                                   ; preds = %for.body29.i.i.preheader, %for.body29.i.i
  %j25.068.i.i = phi i64 [ %inc34.i.i, %for.body29.i.i ], [ %j25.068.i.i.ph, %for.body29.i.i.preheader ]
  %r.167.i.i = phi i64 [ %add32.i.i, %for.body29.i.i ], [ %r.167.i.i.ph, %for.body29.i.i.preheader ]
  %arrayidx30.i.i = getelementptr inbounds i8, i8* %add.ptr.i, i64 %j25.068.i.i
  %135 = load i8, i8* %arrayidx30.i.i, align 1, !tbaa !7, !range !73
  %136 = zext i8 %135 to i64
  %add32.i.i = add nuw nsw i64 %r.167.i.i, %136
  %inc34.i.i = add nuw nsw i64 %j25.068.i.i, 1
  %exitcond73.i.i = icmp eq i64 %inc34.i.i, %sub
  br i1 %exitcond73.i.i, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i, label %for.body29.i.i, !llvm.loop !98

_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i:   ; preds = %for.body29.i.i, %vector.ph452, %middle.block417, %if.else.i.i, %if.then.i.i
  %r.2.i.i = phi i64 [ 0, %if.then.i.i ], [ 0, %if.else.i.i ], [ %134, %middle.block417 ], [ %add21.i.i, %vector.ph452 ], [ %add32.i.i, %for.body29.i.i ]
  %mul.i = shl i64 %r.2.i.i, 3
  %call1.i = tail call noalias i8* @malloc(i64 %mul.i) #2
  %137 = bitcast i8* %call1.i to i64*
  br label %if.end.i

if.end.i:                                         ; preds = %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i, %if.then
  %Out.addr.0.i = phi i64* [ %137, %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i ], [ %Out, %if.then ]
  %cmp220.i = icmp sgt i64 %e, %s
  br i1 %cmp220.i, label %for.body.i.preheader, label %cleanup

for.body.i.preheader:                             ; preds = %if.end.i
  %138 = sub i64 %e, %s
  %139 = add i64 %e, -1
  %140 = sub i64 %139, %s
  %xtraiter = and i64 %138, 3
  %141 = icmp ult i64 %140, 3
  br i1 %141, label %cleanup.loopexit.unr-lcssa, label %for.body.i.preheader.new

for.body.i.preheader.new:                         ; preds = %for.body.i.preheader
  %unroll_iter = sub i64 %138, %xtraiter
  br label %for.body.i

for.body.i:                                       ; preds = %for.inc.i.3, %for.body.i.preheader.new
  %k.022.i = phi i64 [ 0, %for.body.i.preheader.new ], [ %k.1.i.3, %for.inc.i.3 ]
  %storemerge21.i = phi i64 [ %s, %for.body.i.preheader.new ], [ %inc7.i.3, %for.inc.i.3 ]
  %niter = phi i64 [ %unroll_iter, %for.body.i.preheader.new ], [ %niter.nsub.3, %for.inc.i.3 ]
  %arrayidx.i = getelementptr inbounds i8, i8* %Fl, i64 %storemerge21.i
  %142 = load i8, i8* %arrayidx.i, align 1, !tbaa !7, !range !73
  %tobool.i = icmp eq i8 %142, 0
  br i1 %tobool.i, label %for.inc.i, label %if.then3.i

if.then3.i:                                       ; preds = %for.body.i
  %inc.i = add nsw i64 %k.022.i, 1
  %arrayidx5.i = getelementptr inbounds i64, i64* %Out.addr.0.i, i64 %k.022.i
  store i64 %storemerge21.i, i64* %arrayidx5.i, align 8, !tbaa !10
  br label %for.inc.i

for.inc.i:                                        ; preds = %if.then3.i, %for.body.i
  %k.1.i = phi i64 [ %inc.i, %if.then3.i ], [ %k.022.i, %for.body.i ]
  %inc7.i = add nsw i64 %storemerge21.i, 1
  %arrayidx.i.1 = getelementptr inbounds i8, i8* %Fl, i64 %inc7.i
  %143 = load i8, i8* %arrayidx.i.1, align 1, !tbaa !7, !range !73
  %tobool.i.1 = icmp eq i8 %143, 0
  br i1 %tobool.i.1, label %for.inc.i.1, label %if.then3.i.1

pfor.detach.lr.ph:                                ; preds = %entry
  %mul = shl nsw i64 %add, 3
  %call2 = tail call noalias i8* @malloc(i64 %mul) #2
  %144 = bitcast i8* %call2 to i64*
  %145 = icmp sgt i64 %div, 0
  %smax295 = select i1 %145, i64 %div, i64 0
  %146 = xor i64 %s, -1
  %147 = sub i64 -2049, %s
  %148 = xor i64 %e, -1
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %__begin.0290 = phi i64 [ 0, %pfor.detach.lr.ph ], [ %inc, %pfor.inc ]
  %149 = mul i64 %__begin.0290, -2048
  %150 = add i64 %149, %146
  %151 = add i64 %147, %149
  %152 = icmp sgt i64 %151, %148
  %smax533 = select i1 %152, i64 %151, i64 %148
  %153 = sub i64 %150, %smax533
  %154 = add i64 %153, -16
  %155 = lshr i64 %154, 4
  %156 = add nuw nsw i64 %155, 1
  %157 = mul i64 %__begin.0290, -2048
  %158 = add i64 %157, %146
  %159 = add i64 %147, %157
  %160 = icmp sgt i64 %159, %148
  %smax322 = select i1 %160, i64 %159, i64 %148
  %161 = sub i64 %158, %smax322
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad26

pfor.body:                                        ; preds = %pfor.detach
  %mul15 = shl nsw i64 %__begin.0290, 11
  %add16 = add nsw i64 %mul15, %s
  %add18 = add nsw i64 %add16, 2048
  %cmp.i188 = icmp sgt i64 %add18, %e
  %.sroa.speculated = select i1 %cmp.i188, i64 %e, i64 %add18
  %add.ptr = getelementptr inbounds i8, i8* %Fl, i64 %add16
  %sub20 = sub nsw i64 %.sroa.speculated, %add16
  %cmp.i260 = icmp sgt i64 %sub20, 127
  %and.i = and i64 %sub20, 511
  %cmp1.i = icmp eq i64 %and.i, 0
  %or.cond.i = and i1 %cmp.i260, %cmp1.i
  br i1 %or.cond.i, label %land.lhs.true2.i, label %if.else.i

land.lhs.true2.i:                                 ; preds = %pfor.body
  %162 = ptrtoint i8* %add.ptr to i64
  %and3.i = and i64 %162, 3
  %cmp4.i = icmp eq i64 %and3.i, 0
  br i1 %cmp4.i, label %if.then.i261, label %for.body29.lr.ph.i

if.then.i261:                                     ; preds = %land.lhs.true2.i
  %shr75.i = lshr i64 %sub20, 9
  %cmp562.i = icmp sgt i64 %sub20, 511
  br i1 %cmp562.i, label %for.body.lr.ph.i262, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit

for.body.lr.ph.i262:                              ; preds = %if.then.i261
  %163 = bitcast i8* %add.ptr to i32*
  br label %vector.ph

vector.ph:                                        ; preds = %for.body.lr.ph.i262, %vector.ph
  %indvars.iv71.i = phi i64 [ 0, %for.body.lr.ph.i262 ], [ %indvars.iv.next72.i, %vector.ph ]
  %IFl.064.i = phi i32* [ %163, %for.body.lr.ph.i262 ], [ %add.ptr.i264, %vector.ph ]
  %r.063.i = phi i64 [ 0, %for.body.lr.ph.i262 ], [ %add21.i, %vector.ph ]
  %164 = bitcast i32* %IFl.064.i to <8 x i32>*
  %wide.load = load <8 x i32>, <8 x i32>* %164, align 4, !tbaa !55
  %165 = getelementptr i32, i32* %IFl.064.i, i64 8
  %166 = bitcast i32* %165 to <8 x i32>*
  %wide.load309 = load <8 x i32>, <8 x i32>* %166, align 4, !tbaa !55
  %167 = getelementptr i32, i32* %IFl.064.i, i64 16
  %168 = bitcast i32* %167 to <8 x i32>*
  %wide.load310 = load <8 x i32>, <8 x i32>* %168, align 4, !tbaa !55
  %169 = getelementptr i32, i32* %IFl.064.i, i64 24
  %170 = bitcast i32* %169 to <8 x i32>*
  %wide.load311 = load <8 x i32>, <8 x i32>* %170, align 4, !tbaa !55
  %171 = getelementptr inbounds i32, i32* %IFl.064.i, i64 32
  %172 = bitcast i32* %171 to <8 x i32>*
  %wide.load.1 = load <8 x i32>, <8 x i32>* %172, align 4, !tbaa !55
  %173 = getelementptr i32, i32* %IFl.064.i, i64 40
  %174 = bitcast i32* %173 to <8 x i32>*
  %wide.load309.1 = load <8 x i32>, <8 x i32>* %174, align 4, !tbaa !55
  %175 = getelementptr i32, i32* %IFl.064.i, i64 48
  %176 = bitcast i32* %175 to <8 x i32>*
  %wide.load310.1 = load <8 x i32>, <8 x i32>* %176, align 4, !tbaa !55
  %177 = getelementptr i32, i32* %IFl.064.i, i64 56
  %178 = bitcast i32* %177 to <8 x i32>*
  %wide.load311.1 = load <8 x i32>, <8 x i32>* %178, align 4, !tbaa !55
  %179 = add nsw <8 x i32> %wide.load.1, %wide.load
  %180 = add nsw <8 x i32> %wide.load309.1, %wide.load309
  %181 = add nsw <8 x i32> %wide.load310.1, %wide.load310
  %182 = add nsw <8 x i32> %wide.load311.1, %wide.load311
  %183 = getelementptr inbounds i32, i32* %IFl.064.i, i64 64
  %184 = bitcast i32* %183 to <8 x i32>*
  %wide.load.2 = load <8 x i32>, <8 x i32>* %184, align 4, !tbaa !55
  %185 = getelementptr i32, i32* %IFl.064.i, i64 72
  %186 = bitcast i32* %185 to <8 x i32>*
  %wide.load309.2 = load <8 x i32>, <8 x i32>* %186, align 4, !tbaa !55
  %187 = getelementptr i32, i32* %IFl.064.i, i64 80
  %188 = bitcast i32* %187 to <8 x i32>*
  %wide.load310.2 = load <8 x i32>, <8 x i32>* %188, align 4, !tbaa !55
  %189 = getelementptr i32, i32* %IFl.064.i, i64 88
  %190 = bitcast i32* %189 to <8 x i32>*
  %wide.load311.2 = load <8 x i32>, <8 x i32>* %190, align 4, !tbaa !55
  %191 = add nsw <8 x i32> %wide.load.2, %179
  %192 = add nsw <8 x i32> %wide.load309.2, %180
  %193 = add nsw <8 x i32> %wide.load310.2, %181
  %194 = add nsw <8 x i32> %wide.load311.2, %182
  %195 = getelementptr inbounds i32, i32* %IFl.064.i, i64 96
  %196 = bitcast i32* %195 to <8 x i32>*
  %wide.load.3 = load <8 x i32>, <8 x i32>* %196, align 4, !tbaa !55
  %197 = getelementptr i32, i32* %IFl.064.i, i64 104
  %198 = bitcast i32* %197 to <8 x i32>*
  %wide.load309.3 = load <8 x i32>, <8 x i32>* %198, align 4, !tbaa !55
  %199 = getelementptr i32, i32* %IFl.064.i, i64 112
  %200 = bitcast i32* %199 to <8 x i32>*
  %wide.load310.3 = load <8 x i32>, <8 x i32>* %200, align 4, !tbaa !55
  %201 = getelementptr i32, i32* %IFl.064.i, i64 120
  %202 = bitcast i32* %201 to <8 x i32>*
  %wide.load311.3 = load <8 x i32>, <8 x i32>* %202, align 4, !tbaa !55
  %203 = add nsw <8 x i32> %wide.load.3, %191
  %204 = add nsw <8 x i32> %wide.load309.3, %192
  %205 = add nsw <8 x i32> %wide.load310.3, %193
  %206 = add nsw <8 x i32> %wide.load311.3, %194
  %bin.rdx = add <8 x i32> %204, %203
  %bin.rdx312 = add <8 x i32> %205, %bin.rdx
  %bin.rdx313 = add <8 x i32> %206, %bin.rdx312
  %rdx.shuf = shufflevector <8 x i32> %bin.rdx313, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx314 = add <8 x i32> %bin.rdx313, %rdx.shuf
  %rdx.shuf315 = shufflevector <8 x i32> %bin.rdx314, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx316 = add <8 x i32> %bin.rdx314, %rdx.shuf315
  %rdx.shuf317 = shufflevector <8 x i32> %bin.rdx316, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx318 = add <8 x i32> %bin.rdx316, %rdx.shuf317
  %207 = extractelement <8 x i32> %bin.rdx318, i32 0
  %and10.i = and i32 %207, 255
  %208 = lshr i32 %207, 8
  %and12.i = and i32 %208, 255
  %209 = lshr i32 %207, 16
  %and15.i = and i32 %209, 255
  %210 = lshr i32 %207, 24
  %add13.i = add nuw nsw i32 %210, %and10.i
  %add16.i = add nuw nsw i32 %add13.i, %and12.i
  %add19.i = add nuw nsw i32 %add16.i, %and15.i
  %211 = zext i32 %add19.i to i64
  %add21.i = add nuw nsw i64 %r.063.i, %211
  %add.ptr.i264 = getelementptr inbounds i32, i32* %IFl.064.i, i64 128
  %indvars.iv.next72.i = add nuw nsw i64 %indvars.iv71.i, 1
  %cmp5.i = icmp ugt i64 %shr75.i, %indvars.iv.next72.i
  br i1 %cmp5.i, label %vector.ph, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit

if.else.i:                                        ; preds = %pfor.body
  %cmp2766.i = icmp sgt i64 %sub20, 0
  br i1 %cmp2766.i, label %for.body29.lr.ph.i, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit

for.body29.lr.ph.i:                               ; preds = %if.else.i, %land.lhs.true2.i
  %min.iters.check = icmp ult i64 %161, 16
  br i1 %min.iters.check, label %for.body29.i.preheader, label %vector.ph323

vector.ph323:                                     ; preds = %for.body29.lr.ph.i
  %n.vec = and i64 %161, -16
  %xtraiter534 = and i64 %156, 3
  %212 = icmp ult i64 %154, 48
  br i1 %212, label %middle.block320.unr-lcssa, label %vector.ph323.new

vector.ph323.new:                                 ; preds = %vector.ph323
  %unroll_iter541 = sub nsw i64 %156, %xtraiter534
  br label %vector.body319

vector.body319:                                   ; preds = %vector.body319, %vector.ph323.new
  %index324 = phi i64 [ 0, %vector.ph323.new ], [ %index.next325.3, %vector.body319 ]
  %vec.phi334 = phi <4 x i64> [ zeroinitializer, %vector.ph323.new ], [ %273, %vector.body319 ]
  %vec.phi335 = phi <4 x i64> [ zeroinitializer, %vector.ph323.new ], [ %274, %vector.body319 ]
  %vec.phi336 = phi <4 x i64> [ zeroinitializer, %vector.ph323.new ], [ %275, %vector.body319 ]
  %vec.phi337 = phi <4 x i64> [ zeroinitializer, %vector.ph323.new ], [ %276, %vector.body319 ]
  %niter542 = phi i64 [ %unroll_iter541, %vector.ph323.new ], [ %niter542.nsub.3, %vector.body319 ]
  %213 = getelementptr inbounds i8, i8* %add.ptr, i64 %index324
  %214 = bitcast i8* %213 to <4 x i8>*
  %wide.load338 = load <4 x i8>, <4 x i8>* %214, align 1, !tbaa !7
  %215 = getelementptr i8, i8* %213, i64 4
  %216 = bitcast i8* %215 to <4 x i8>*
  %wide.load339 = load <4 x i8>, <4 x i8>* %216, align 1, !tbaa !7
  %217 = getelementptr i8, i8* %213, i64 8
  %218 = bitcast i8* %217 to <4 x i8>*
  %wide.load340 = load <4 x i8>, <4 x i8>* %218, align 1, !tbaa !7
  %219 = getelementptr i8, i8* %213, i64 12
  %220 = bitcast i8* %219 to <4 x i8>*
  %wide.load341 = load <4 x i8>, <4 x i8>* %220, align 1, !tbaa !7
  %221 = zext <4 x i8> %wide.load338 to <4 x i64>
  %222 = zext <4 x i8> %wide.load339 to <4 x i64>
  %223 = zext <4 x i8> %wide.load340 to <4 x i64>
  %224 = zext <4 x i8> %wide.load341 to <4 x i64>
  %225 = add nuw nsw <4 x i64> %vec.phi334, %221
  %226 = add nuw nsw <4 x i64> %vec.phi335, %222
  %227 = add nuw nsw <4 x i64> %vec.phi336, %223
  %228 = add nuw nsw <4 x i64> %vec.phi337, %224
  %index.next325 = or i64 %index324, 16
  %229 = getelementptr inbounds i8, i8* %add.ptr, i64 %index.next325
  %230 = bitcast i8* %229 to <4 x i8>*
  %wide.load338.1 = load <4 x i8>, <4 x i8>* %230, align 1, !tbaa !7
  %231 = getelementptr i8, i8* %229, i64 4
  %232 = bitcast i8* %231 to <4 x i8>*
  %wide.load339.1 = load <4 x i8>, <4 x i8>* %232, align 1, !tbaa !7
  %233 = getelementptr i8, i8* %229, i64 8
  %234 = bitcast i8* %233 to <4 x i8>*
  %wide.load340.1 = load <4 x i8>, <4 x i8>* %234, align 1, !tbaa !7
  %235 = getelementptr i8, i8* %229, i64 12
  %236 = bitcast i8* %235 to <4 x i8>*
  %wide.load341.1 = load <4 x i8>, <4 x i8>* %236, align 1, !tbaa !7
  %237 = zext <4 x i8> %wide.load338.1 to <4 x i64>
  %238 = zext <4 x i8> %wide.load339.1 to <4 x i64>
  %239 = zext <4 x i8> %wide.load340.1 to <4 x i64>
  %240 = zext <4 x i8> %wide.load341.1 to <4 x i64>
  %241 = add nuw nsw <4 x i64> %225, %237
  %242 = add nuw nsw <4 x i64> %226, %238
  %243 = add nuw nsw <4 x i64> %227, %239
  %244 = add nuw nsw <4 x i64> %228, %240
  %index.next325.1 = or i64 %index324, 32
  %245 = getelementptr inbounds i8, i8* %add.ptr, i64 %index.next325.1
  %246 = bitcast i8* %245 to <4 x i8>*
  %wide.load338.2 = load <4 x i8>, <4 x i8>* %246, align 1, !tbaa !7
  %247 = getelementptr i8, i8* %245, i64 4
  %248 = bitcast i8* %247 to <4 x i8>*
  %wide.load339.2 = load <4 x i8>, <4 x i8>* %248, align 1, !tbaa !7
  %249 = getelementptr i8, i8* %245, i64 8
  %250 = bitcast i8* %249 to <4 x i8>*
  %wide.load340.2 = load <4 x i8>, <4 x i8>* %250, align 1, !tbaa !7
  %251 = getelementptr i8, i8* %245, i64 12
  %252 = bitcast i8* %251 to <4 x i8>*
  %wide.load341.2 = load <4 x i8>, <4 x i8>* %252, align 1, !tbaa !7
  %253 = zext <4 x i8> %wide.load338.2 to <4 x i64>
  %254 = zext <4 x i8> %wide.load339.2 to <4 x i64>
  %255 = zext <4 x i8> %wide.load340.2 to <4 x i64>
  %256 = zext <4 x i8> %wide.load341.2 to <4 x i64>
  %257 = add nuw nsw <4 x i64> %241, %253
  %258 = add nuw nsw <4 x i64> %242, %254
  %259 = add nuw nsw <4 x i64> %243, %255
  %260 = add nuw nsw <4 x i64> %244, %256
  %index.next325.2 = or i64 %index324, 48
  %261 = getelementptr inbounds i8, i8* %add.ptr, i64 %index.next325.2
  %262 = bitcast i8* %261 to <4 x i8>*
  %wide.load338.3 = load <4 x i8>, <4 x i8>* %262, align 1, !tbaa !7
  %263 = getelementptr i8, i8* %261, i64 4
  %264 = bitcast i8* %263 to <4 x i8>*
  %wide.load339.3 = load <4 x i8>, <4 x i8>* %264, align 1, !tbaa !7
  %265 = getelementptr i8, i8* %261, i64 8
  %266 = bitcast i8* %265 to <4 x i8>*
  %wide.load340.3 = load <4 x i8>, <4 x i8>* %266, align 1, !tbaa !7
  %267 = getelementptr i8, i8* %261, i64 12
  %268 = bitcast i8* %267 to <4 x i8>*
  %wide.load341.3 = load <4 x i8>, <4 x i8>* %268, align 1, !tbaa !7
  %269 = zext <4 x i8> %wide.load338.3 to <4 x i64>
  %270 = zext <4 x i8> %wide.load339.3 to <4 x i64>
  %271 = zext <4 x i8> %wide.load340.3 to <4 x i64>
  %272 = zext <4 x i8> %wide.load341.3 to <4 x i64>
  %273 = add nuw nsw <4 x i64> %257, %269
  %274 = add nuw nsw <4 x i64> %258, %270
  %275 = add nuw nsw <4 x i64> %259, %271
  %276 = add nuw nsw <4 x i64> %260, %272
  %index.next325.3 = add i64 %index324, 64
  %niter542.nsub.3 = add i64 %niter542, -4
  %niter542.ncmp.3 = icmp eq i64 %niter542.nsub.3, 0
  br i1 %niter542.ncmp.3, label %middle.block320.unr-lcssa, label %vector.body319, !llvm.loop !99

middle.block320.unr-lcssa:                        ; preds = %vector.body319, %vector.ph323
  %.lcssa502.ph = phi <4 x i64> [ undef, %vector.ph323 ], [ %273, %vector.body319 ]
  %.lcssa501.ph = phi <4 x i64> [ undef, %vector.ph323 ], [ %274, %vector.body319 ]
  %.lcssa500.ph = phi <4 x i64> [ undef, %vector.ph323 ], [ %275, %vector.body319 ]
  %.lcssa499.ph = phi <4 x i64> [ undef, %vector.ph323 ], [ %276, %vector.body319 ]
  %index324.unr = phi i64 [ 0, %vector.ph323 ], [ %index.next325.3, %vector.body319 ]
  %vec.phi334.unr = phi <4 x i64> [ zeroinitializer, %vector.ph323 ], [ %273, %vector.body319 ]
  %vec.phi335.unr = phi <4 x i64> [ zeroinitializer, %vector.ph323 ], [ %274, %vector.body319 ]
  %vec.phi336.unr = phi <4 x i64> [ zeroinitializer, %vector.ph323 ], [ %275, %vector.body319 ]
  %vec.phi337.unr = phi <4 x i64> [ zeroinitializer, %vector.ph323 ], [ %276, %vector.body319 ]
  %lcmp.mod536 = icmp eq i64 %xtraiter534, 0
  br i1 %lcmp.mod536, label %middle.block320, label %vector.body319.epil.preheader

vector.body319.epil.preheader:                    ; preds = %middle.block320.unr-lcssa
  br label %vector.body319.epil

vector.body319.epil:                              ; preds = %vector.body319.epil, %vector.body319.epil.preheader
  %index324.epil = phi i64 [ %index324.unr, %vector.body319.epil.preheader ], [ %index.next325.epil, %vector.body319.epil ]
  %vec.phi334.epil = phi <4 x i64> [ %vec.phi334.unr, %vector.body319.epil.preheader ], [ %289, %vector.body319.epil ]
  %vec.phi335.epil = phi <4 x i64> [ %vec.phi335.unr, %vector.body319.epil.preheader ], [ %290, %vector.body319.epil ]
  %vec.phi336.epil = phi <4 x i64> [ %vec.phi336.unr, %vector.body319.epil.preheader ], [ %291, %vector.body319.epil ]
  %vec.phi337.epil = phi <4 x i64> [ %vec.phi337.unr, %vector.body319.epil.preheader ], [ %292, %vector.body319.epil ]
  %epil.iter535 = phi i64 [ %xtraiter534, %vector.body319.epil.preheader ], [ %epil.iter535.sub, %vector.body319.epil ]
  %277 = getelementptr inbounds i8, i8* %add.ptr, i64 %index324.epil
  %278 = bitcast i8* %277 to <4 x i8>*
  %wide.load338.epil = load <4 x i8>, <4 x i8>* %278, align 1, !tbaa !7
  %279 = getelementptr i8, i8* %277, i64 4
  %280 = bitcast i8* %279 to <4 x i8>*
  %wide.load339.epil = load <4 x i8>, <4 x i8>* %280, align 1, !tbaa !7
  %281 = getelementptr i8, i8* %277, i64 8
  %282 = bitcast i8* %281 to <4 x i8>*
  %wide.load340.epil = load <4 x i8>, <4 x i8>* %282, align 1, !tbaa !7
  %283 = getelementptr i8, i8* %277, i64 12
  %284 = bitcast i8* %283 to <4 x i8>*
  %wide.load341.epil = load <4 x i8>, <4 x i8>* %284, align 1, !tbaa !7
  %285 = zext <4 x i8> %wide.load338.epil to <4 x i64>
  %286 = zext <4 x i8> %wide.load339.epil to <4 x i64>
  %287 = zext <4 x i8> %wide.load340.epil to <4 x i64>
  %288 = zext <4 x i8> %wide.load341.epil to <4 x i64>
  %289 = add nuw nsw <4 x i64> %vec.phi334.epil, %285
  %290 = add nuw nsw <4 x i64> %vec.phi335.epil, %286
  %291 = add nuw nsw <4 x i64> %vec.phi336.epil, %287
  %292 = add nuw nsw <4 x i64> %vec.phi337.epil, %288
  %index.next325.epil = add i64 %index324.epil, 16
  %epil.iter535.sub = add i64 %epil.iter535, -1
  %epil.iter535.cmp = icmp eq i64 %epil.iter535.sub, 0
  br i1 %epil.iter535.cmp, label %middle.block320, label %vector.body319.epil, !llvm.loop !100

middle.block320:                                  ; preds = %vector.body319.epil, %middle.block320.unr-lcssa
  %.lcssa502 = phi <4 x i64> [ %.lcssa502.ph, %middle.block320.unr-lcssa ], [ %289, %vector.body319.epil ]
  %.lcssa501 = phi <4 x i64> [ %.lcssa501.ph, %middle.block320.unr-lcssa ], [ %290, %vector.body319.epil ]
  %.lcssa500 = phi <4 x i64> [ %.lcssa500.ph, %middle.block320.unr-lcssa ], [ %291, %vector.body319.epil ]
  %.lcssa499 = phi <4 x i64> [ %.lcssa499.ph, %middle.block320.unr-lcssa ], [ %292, %vector.body319.epil ]
  %bin.rdx342 = add nuw <4 x i64> %.lcssa501, %.lcssa502
  %bin.rdx343 = add <4 x i64> %.lcssa500, %bin.rdx342
  %bin.rdx344 = add <4 x i64> %.lcssa499, %bin.rdx343
  %rdx.shuf345 = shufflevector <4 x i64> %bin.rdx344, <4 x i64> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx346 = add <4 x i64> %bin.rdx344, %rdx.shuf345
  %rdx.shuf347 = shufflevector <4 x i64> %bin.rdx346, <4 x i64> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx348 = add <4 x i64> %bin.rdx346, %rdx.shuf347
  %293 = extractelement <4 x i64> %bin.rdx348, i32 0
  %cmp.n327 = icmp eq i64 %161, %n.vec
  br i1 %cmp.n327, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit, label %for.body29.i.preheader

for.body29.i.preheader:                           ; preds = %middle.block320, %for.body29.lr.ph.i
  %j25.068.i.ph = phi i64 [ 0, %for.body29.lr.ph.i ], [ %n.vec, %middle.block320 ]
  %r.167.i.ph = phi i64 [ 0, %for.body29.lr.ph.i ], [ %293, %middle.block320 ]
  br label %for.body29.i

for.body29.i:                                     ; preds = %for.body29.i.preheader, %for.body29.i
  %j25.068.i = phi i64 [ %inc34.i, %for.body29.i ], [ %j25.068.i.ph, %for.body29.i.preheader ]
  %r.167.i = phi i64 [ %add32.i, %for.body29.i ], [ %r.167.i.ph, %for.body29.i.preheader ]
  %arrayidx30.i = getelementptr inbounds i8, i8* %add.ptr, i64 %j25.068.i
  %294 = load i8, i8* %arrayidx30.i, align 1, !tbaa !7, !range !73
  %295 = zext i8 %294 to i64
  %add32.i = add nuw nsw i64 %r.167.i, %295
  %inc34.i = add nuw nsw i64 %j25.068.i, 1
  %exitcond73.i = icmp eq i64 %inc34.i, %sub20
  br i1 %exitcond73.i, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit, label %for.body29.i, !llvm.loop !101

_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit:     ; preds = %for.body29.i, %vector.ph, %middle.block320, %if.then.i261, %if.else.i
  %r.2.i = phi i64 [ 0, %if.then.i261 ], [ 0, %if.else.i ], [ %293, %middle.block320 ], [ %add21.i, %vector.ph ], [ %add32.i, %for.body29.i ]
  %arrayidx = getelementptr inbounds i64, i64* %144, i64 %__begin.0290
  store i64 %r.2.i, i64* %arrayidx, align 8, !tbaa !10
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.detach, %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit
  %inc = add nuw nsw i64 %__begin.0290, 1
  %exitcond296 = icmp eq i64 %__begin.0290, %smax295
  br i1 %exitcond296, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !102

lpad26:                                           ; preds = %pfor.detach
  %296 = landingpad { i8*, i32 }
          cleanup
  %297 = extractvalue { i8*, i32 } %296, 0
  %298 = extractvalue { i8*, i32 } %296, 1
  sync within %syncreg, label %ehcleanup113

sync.continue:                                    ; preds = %pfor.cond.cleanup
  %call.i = tail call i64 @_ZN8sequence4scanIllN5utils4addFIlEENS_4getAIllEEEET_PS6_T0_S8_T1_T2_S6_bb(i64* %144, i64 0, i64 %add, i64* %144, i64 0, i1 zeroext false, i1 zeroext false)
  %cmp40 = icmp eq i64* %Out, null
  br i1 %cmp40, label %if.then41, label %if.end44

if.then41:                                        ; preds = %sync.continue
  %mul42 = shl i64 %call.i, 3
  %call43 = tail call noalias i8* @malloc(i64 %mul42) #2
  %299 = bitcast i8* %call43 to i64*
  br label %if.end44

if.end44:                                         ; preds = %if.then41, %sync.continue
  %Out.addr.0 = phi i64* [ %299, %if.then41 ], [ %Out, %sync.continue ]
  %cmp62287 = icmp slt i64 %sub, -2046
  br i1 %cmp62287, label %pfor.cond.cleanup63, label %pfor.detach64.lr.ph

pfor.detach64.lr.ph:                              ; preds = %if.end44
  %300 = icmp sgt i64 %div, 0
  %smax = select i1 %300, i64 %div, i64 0
  %301 = xor i64 %s, -1
  %302 = sub i64 -2049, %s
  %303 = xor i64 %e, -1
  %304 = sub i64 -2, %s
  br label %pfor.detach64

pfor.cond.cleanup63:                              ; preds = %pfor.inc92, %if.end44
  sync within %syncreg53, label %sync.continue101

pfor.detach64:                                    ; preds = %pfor.inc92, %pfor.detach64.lr.ph
  %__begin55.0288 = phi i64 [ 0, %pfor.detach64.lr.ph ], [ %inc93, %pfor.inc92 ]
  %305 = mul i64 %__begin55.0288, -2048
  %306 = add i64 %305, %301
  %307 = add i64 %302, %305
  %308 = icmp sgt i64 %307, %303
  %smax527 = select i1 %308, i64 %307, i64 %303
  %309 = sub i64 %306, %smax527
  %310 = add i64 %304, %305
  %311 = sub i64 %310, %smax527
  %312 = mul i64 %__begin55.0288, -2048
  %313 = add i64 %312, %301
  %314 = add i64 %302, %312
  %315 = icmp sgt i64 %314, %303
  %smax517 = select i1 %315, i64 %314, i64 %303
  %316 = sub i64 %313, %smax517
  %317 = add i64 %316, -16
  %318 = lshr i64 %317, 4
  %319 = add nuw nsw i64 %318, 1
  %320 = mul i64 %__begin55.0288, -2048
  %321 = add i64 %320, %301
  %322 = add i64 %302, %320
  %323 = icmp sgt i64 %322, %303
  %smax385 = select i1 %323, i64 %322, i64 %303
  %324 = sub i64 %321, %smax385
  detach within %syncreg53, label %pfor.body69, label %pfor.inc92 unwind label %lpad94

pfor.body69:                                      ; preds = %pfor.detach64
  %mul71 = shl nsw i64 %__begin55.0288, 11
  %add72 = add nsw i64 %mul71, %s
  %add75 = add nsw i64 %add72, 2048
  %cmp.i268 = icmp sgt i64 %add75, %e
  %.sroa.speculated275 = select i1 %cmp.i268, i64 %e, i64 %add75
  %arrayidx82 = getelementptr inbounds i64, i64* %144, i64 %__begin55.0288
  %325 = load i64, i64* %arrayidx82, align 8, !tbaa !10
  %add.ptr83 = getelementptr inbounds i64, i64* %Out.addr.0, i64 %325
  %cmp.i189 = icmp eq i64* %add.ptr83, null
  br i1 %cmp.i189, label %if.then.i196, label %if.end.i242

if.then.i196:                                     ; preds = %pfor.body69
  %add.ptr.i190 = getelementptr inbounds i8, i8* %Fl, i64 %add72
  %sub.i191 = sub nsw i64 %.sroa.speculated275, %add72
  %cmp.i.i192 = icmp sgt i64 %sub.i191, 127
  %and.i.i193 = and i64 %sub.i191, 511
  %cmp1.i.i194 = icmp eq i64 %and.i.i193, 0
  %or.cond.i.i195 = and i1 %cmp.i.i192, %cmp1.i.i194
  br i1 %or.cond.i.i195, label %land.lhs.true2.i.i199, label %if.else.i.i227

land.lhs.true2.i.i199:                            ; preds = %if.then.i196
  %326 = ptrtoint i8* %add.ptr.i190 to i64
  %and3.i.i197 = and i64 %326, 3
  %cmp4.i.i198 = icmp eq i64 %and3.i.i197, 0
  br i1 %cmp4.i.i198, label %if.then.i.i202, label %for.body29.lr.ph.i.i228

if.then.i.i202:                                   ; preds = %land.lhs.true2.i.i199
  %shr75.i.i200 = lshr i64 %sub.i191, 9
  %cmp562.i.i201 = icmp sgt i64 %sub.i191, 511
  br i1 %cmp562.i.i201, label %for.body.lr.ph.i.i203, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i239

for.body.lr.ph.i.i203:                            ; preds = %if.then.i.i202
  %327 = bitcast i8* %add.ptr.i190 to i32*
  br label %vector.ph353

vector.ph353:                                     ; preds = %for.body.lr.ph.i.i203, %vector.ph353
  %indvars.iv71.i.i204 = phi i64 [ 0, %for.body.lr.ph.i.i203 ], [ %indvars.iv.next72.i.i216, %vector.ph353 ]
  %IFl.064.i.i205 = phi i32* [ %327, %for.body.lr.ph.i.i203 ], [ %add.ptr.i.i215, %vector.ph353 ]
  %r.063.i.i206 = phi i64 [ 0, %for.body.lr.ph.i.i203 ], [ %add21.i.i214, %vector.ph353 ]
  %328 = bitcast i32* %IFl.064.i.i205 to <8 x i32>*
  %wide.load368 = load <8 x i32>, <8 x i32>* %328, align 4, !tbaa !55
  %329 = getelementptr i32, i32* %IFl.064.i.i205, i64 8
  %330 = bitcast i32* %329 to <8 x i32>*
  %wide.load369 = load <8 x i32>, <8 x i32>* %330, align 4, !tbaa !55
  %331 = getelementptr i32, i32* %IFl.064.i.i205, i64 16
  %332 = bitcast i32* %331 to <8 x i32>*
  %wide.load370 = load <8 x i32>, <8 x i32>* %332, align 4, !tbaa !55
  %333 = getelementptr i32, i32* %IFl.064.i.i205, i64 24
  %334 = bitcast i32* %333 to <8 x i32>*
  %wide.load371 = load <8 x i32>, <8 x i32>* %334, align 4, !tbaa !55
  %335 = getelementptr inbounds i32, i32* %IFl.064.i.i205, i64 32
  %336 = bitcast i32* %335 to <8 x i32>*
  %wide.load368.1 = load <8 x i32>, <8 x i32>* %336, align 4, !tbaa !55
  %337 = getelementptr i32, i32* %IFl.064.i.i205, i64 40
  %338 = bitcast i32* %337 to <8 x i32>*
  %wide.load369.1 = load <8 x i32>, <8 x i32>* %338, align 4, !tbaa !55
  %339 = getelementptr i32, i32* %IFl.064.i.i205, i64 48
  %340 = bitcast i32* %339 to <8 x i32>*
  %wide.load370.1 = load <8 x i32>, <8 x i32>* %340, align 4, !tbaa !55
  %341 = getelementptr i32, i32* %IFl.064.i.i205, i64 56
  %342 = bitcast i32* %341 to <8 x i32>*
  %wide.load371.1 = load <8 x i32>, <8 x i32>* %342, align 4, !tbaa !55
  %343 = add nsw <8 x i32> %wide.load368.1, %wide.load368
  %344 = add nsw <8 x i32> %wide.load369.1, %wide.load369
  %345 = add nsw <8 x i32> %wide.load370.1, %wide.load370
  %346 = add nsw <8 x i32> %wide.load371.1, %wide.load371
  %347 = getelementptr inbounds i32, i32* %IFl.064.i.i205, i64 64
  %348 = bitcast i32* %347 to <8 x i32>*
  %wide.load368.2 = load <8 x i32>, <8 x i32>* %348, align 4, !tbaa !55
  %349 = getelementptr i32, i32* %IFl.064.i.i205, i64 72
  %350 = bitcast i32* %349 to <8 x i32>*
  %wide.load369.2 = load <8 x i32>, <8 x i32>* %350, align 4, !tbaa !55
  %351 = getelementptr i32, i32* %IFl.064.i.i205, i64 80
  %352 = bitcast i32* %351 to <8 x i32>*
  %wide.load370.2 = load <8 x i32>, <8 x i32>* %352, align 4, !tbaa !55
  %353 = getelementptr i32, i32* %IFl.064.i.i205, i64 88
  %354 = bitcast i32* %353 to <8 x i32>*
  %wide.load371.2 = load <8 x i32>, <8 x i32>* %354, align 4, !tbaa !55
  %355 = add nsw <8 x i32> %wide.load368.2, %343
  %356 = add nsw <8 x i32> %wide.load369.2, %344
  %357 = add nsw <8 x i32> %wide.load370.2, %345
  %358 = add nsw <8 x i32> %wide.load371.2, %346
  %359 = getelementptr inbounds i32, i32* %IFl.064.i.i205, i64 96
  %360 = bitcast i32* %359 to <8 x i32>*
  %wide.load368.3 = load <8 x i32>, <8 x i32>* %360, align 4, !tbaa !55
  %361 = getelementptr i32, i32* %IFl.064.i.i205, i64 104
  %362 = bitcast i32* %361 to <8 x i32>*
  %wide.load369.3 = load <8 x i32>, <8 x i32>* %362, align 4, !tbaa !55
  %363 = getelementptr i32, i32* %IFl.064.i.i205, i64 112
  %364 = bitcast i32* %363 to <8 x i32>*
  %wide.load370.3 = load <8 x i32>, <8 x i32>* %364, align 4, !tbaa !55
  %365 = getelementptr i32, i32* %IFl.064.i.i205, i64 120
  %366 = bitcast i32* %365 to <8 x i32>*
  %wide.load371.3 = load <8 x i32>, <8 x i32>* %366, align 4, !tbaa !55
  %367 = add nsw <8 x i32> %wide.load368.3, %355
  %368 = add nsw <8 x i32> %wide.load369.3, %356
  %369 = add nsw <8 x i32> %wide.load370.3, %357
  %370 = add nsw <8 x i32> %wide.load371.3, %358
  %bin.rdx372 = add <8 x i32> %368, %367
  %bin.rdx373 = add <8 x i32> %369, %bin.rdx372
  %bin.rdx374 = add <8 x i32> %370, %bin.rdx373
  %rdx.shuf375 = shufflevector <8 x i32> %bin.rdx374, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx376 = add <8 x i32> %bin.rdx374, %rdx.shuf375
  %rdx.shuf377 = shufflevector <8 x i32> %bin.rdx376, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx378 = add <8 x i32> %bin.rdx376, %rdx.shuf377
  %rdx.shuf379 = shufflevector <8 x i32> %bin.rdx378, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx380 = add <8 x i32> %bin.rdx378, %rdx.shuf379
  %371 = extractelement <8 x i32> %bin.rdx380, i32 0
  %and10.i.i208 = and i32 %371, 255
  %372 = lshr i32 %371, 8
  %and12.i.i209 = and i32 %372, 255
  %373 = lshr i32 %371, 16
  %and15.i.i210 = and i32 %373, 255
  %374 = lshr i32 %371, 24
  %add13.i.i211 = add nuw nsw i32 %374, %and10.i.i208
  %add16.i.i212 = add nuw nsw i32 %add13.i.i211, %and12.i.i209
  %add19.i.i213 = add nuw nsw i32 %add16.i.i212, %and15.i.i210
  %375 = zext i32 %add19.i.i213 to i64
  %add21.i.i214 = add nuw nsw i64 %r.063.i.i206, %375
  %add.ptr.i.i215 = getelementptr inbounds i32, i32* %IFl.064.i.i205, i64 128
  %indvars.iv.next72.i.i216 = add nuw nsw i64 %indvars.iv71.i.i204, 1
  %cmp5.i.i217 = icmp ugt i64 %shr75.i.i200, %indvars.iv.next72.i.i216
  br i1 %cmp5.i.i217, label %vector.ph353, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i239

if.else.i.i227:                                   ; preds = %if.then.i196
  %cmp2766.i.i226 = icmp sgt i64 %sub.i191, 0
  br i1 %cmp2766.i.i226, label %for.body29.lr.ph.i.i228, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i239

for.body29.lr.ph.i.i228:                          ; preds = %if.else.i.i227, %land.lhs.true2.i.i199
  %min.iters.check386 = icmp ult i64 %324, 16
  br i1 %min.iters.check386, label %for.body29.i.i235.preheader, label %vector.ph387

vector.ph387:                                     ; preds = %for.body29.lr.ph.i.i228
  %n.vec389 = and i64 %324, -16
  %xtraiter518 = and i64 %319, 3
  %376 = icmp ult i64 %317, 48
  br i1 %376, label %middle.block383.unr-lcssa, label %vector.ph387.new

vector.ph387.new:                                 ; preds = %vector.ph387
  %unroll_iter525 = sub nsw i64 %319, %xtraiter518
  br label %vector.body382

vector.body382:                                   ; preds = %vector.body382, %vector.ph387.new
  %index390 = phi i64 [ 0, %vector.ph387.new ], [ %index.next391.3, %vector.body382 ]
  %vec.phi400 = phi <4 x i64> [ zeroinitializer, %vector.ph387.new ], [ %437, %vector.body382 ]
  %vec.phi401 = phi <4 x i64> [ zeroinitializer, %vector.ph387.new ], [ %438, %vector.body382 ]
  %vec.phi402 = phi <4 x i64> [ zeroinitializer, %vector.ph387.new ], [ %439, %vector.body382 ]
  %vec.phi403 = phi <4 x i64> [ zeroinitializer, %vector.ph387.new ], [ %440, %vector.body382 ]
  %niter526 = phi i64 [ %unroll_iter525, %vector.ph387.new ], [ %niter526.nsub.3, %vector.body382 ]
  %377 = getelementptr inbounds i8, i8* %add.ptr.i190, i64 %index390
  %378 = bitcast i8* %377 to <4 x i8>*
  %wide.load404 = load <4 x i8>, <4 x i8>* %378, align 1, !tbaa !7
  %379 = getelementptr i8, i8* %377, i64 4
  %380 = bitcast i8* %379 to <4 x i8>*
  %wide.load405 = load <4 x i8>, <4 x i8>* %380, align 1, !tbaa !7
  %381 = getelementptr i8, i8* %377, i64 8
  %382 = bitcast i8* %381 to <4 x i8>*
  %wide.load406 = load <4 x i8>, <4 x i8>* %382, align 1, !tbaa !7
  %383 = getelementptr i8, i8* %377, i64 12
  %384 = bitcast i8* %383 to <4 x i8>*
  %wide.load407 = load <4 x i8>, <4 x i8>* %384, align 1, !tbaa !7
  %385 = zext <4 x i8> %wide.load404 to <4 x i64>
  %386 = zext <4 x i8> %wide.load405 to <4 x i64>
  %387 = zext <4 x i8> %wide.load406 to <4 x i64>
  %388 = zext <4 x i8> %wide.load407 to <4 x i64>
  %389 = add nuw nsw <4 x i64> %vec.phi400, %385
  %390 = add nuw nsw <4 x i64> %vec.phi401, %386
  %391 = add nuw nsw <4 x i64> %vec.phi402, %387
  %392 = add nuw nsw <4 x i64> %vec.phi403, %388
  %index.next391 = or i64 %index390, 16
  %393 = getelementptr inbounds i8, i8* %add.ptr.i190, i64 %index.next391
  %394 = bitcast i8* %393 to <4 x i8>*
  %wide.load404.1 = load <4 x i8>, <4 x i8>* %394, align 1, !tbaa !7
  %395 = getelementptr i8, i8* %393, i64 4
  %396 = bitcast i8* %395 to <4 x i8>*
  %wide.load405.1 = load <4 x i8>, <4 x i8>* %396, align 1, !tbaa !7
  %397 = getelementptr i8, i8* %393, i64 8
  %398 = bitcast i8* %397 to <4 x i8>*
  %wide.load406.1 = load <4 x i8>, <4 x i8>* %398, align 1, !tbaa !7
  %399 = getelementptr i8, i8* %393, i64 12
  %400 = bitcast i8* %399 to <4 x i8>*
  %wide.load407.1 = load <4 x i8>, <4 x i8>* %400, align 1, !tbaa !7
  %401 = zext <4 x i8> %wide.load404.1 to <4 x i64>
  %402 = zext <4 x i8> %wide.load405.1 to <4 x i64>
  %403 = zext <4 x i8> %wide.load406.1 to <4 x i64>
  %404 = zext <4 x i8> %wide.load407.1 to <4 x i64>
  %405 = add nuw nsw <4 x i64> %389, %401
  %406 = add nuw nsw <4 x i64> %390, %402
  %407 = add nuw nsw <4 x i64> %391, %403
  %408 = add nuw nsw <4 x i64> %392, %404
  %index.next391.1 = or i64 %index390, 32
  %409 = getelementptr inbounds i8, i8* %add.ptr.i190, i64 %index.next391.1
  %410 = bitcast i8* %409 to <4 x i8>*
  %wide.load404.2 = load <4 x i8>, <4 x i8>* %410, align 1, !tbaa !7
  %411 = getelementptr i8, i8* %409, i64 4
  %412 = bitcast i8* %411 to <4 x i8>*
  %wide.load405.2 = load <4 x i8>, <4 x i8>* %412, align 1, !tbaa !7
  %413 = getelementptr i8, i8* %409, i64 8
  %414 = bitcast i8* %413 to <4 x i8>*
  %wide.load406.2 = load <4 x i8>, <4 x i8>* %414, align 1, !tbaa !7
  %415 = getelementptr i8, i8* %409, i64 12
  %416 = bitcast i8* %415 to <4 x i8>*
  %wide.load407.2 = load <4 x i8>, <4 x i8>* %416, align 1, !tbaa !7
  %417 = zext <4 x i8> %wide.load404.2 to <4 x i64>
  %418 = zext <4 x i8> %wide.load405.2 to <4 x i64>
  %419 = zext <4 x i8> %wide.load406.2 to <4 x i64>
  %420 = zext <4 x i8> %wide.load407.2 to <4 x i64>
  %421 = add nuw nsw <4 x i64> %405, %417
  %422 = add nuw nsw <4 x i64> %406, %418
  %423 = add nuw nsw <4 x i64> %407, %419
  %424 = add nuw nsw <4 x i64> %408, %420
  %index.next391.2 = or i64 %index390, 48
  %425 = getelementptr inbounds i8, i8* %add.ptr.i190, i64 %index.next391.2
  %426 = bitcast i8* %425 to <4 x i8>*
  %wide.load404.3 = load <4 x i8>, <4 x i8>* %426, align 1, !tbaa !7
  %427 = getelementptr i8, i8* %425, i64 4
  %428 = bitcast i8* %427 to <4 x i8>*
  %wide.load405.3 = load <4 x i8>, <4 x i8>* %428, align 1, !tbaa !7
  %429 = getelementptr i8, i8* %425, i64 8
  %430 = bitcast i8* %429 to <4 x i8>*
  %wide.load406.3 = load <4 x i8>, <4 x i8>* %430, align 1, !tbaa !7
  %431 = getelementptr i8, i8* %425, i64 12
  %432 = bitcast i8* %431 to <4 x i8>*
  %wide.load407.3 = load <4 x i8>, <4 x i8>* %432, align 1, !tbaa !7
  %433 = zext <4 x i8> %wide.load404.3 to <4 x i64>
  %434 = zext <4 x i8> %wide.load405.3 to <4 x i64>
  %435 = zext <4 x i8> %wide.load406.3 to <4 x i64>
  %436 = zext <4 x i8> %wide.load407.3 to <4 x i64>
  %437 = add nuw nsw <4 x i64> %421, %433
  %438 = add nuw nsw <4 x i64> %422, %434
  %439 = add nuw nsw <4 x i64> %423, %435
  %440 = add nuw nsw <4 x i64> %424, %436
  %index.next391.3 = add i64 %index390, 64
  %niter526.nsub.3 = add i64 %niter526, -4
  %niter526.ncmp.3 = icmp eq i64 %niter526.nsub.3, 0
  br i1 %niter526.ncmp.3, label %middle.block383.unr-lcssa, label %vector.body382, !llvm.loop !103

middle.block383.unr-lcssa:                        ; preds = %vector.body382, %vector.ph387
  %.lcssa494.ph = phi <4 x i64> [ undef, %vector.ph387 ], [ %437, %vector.body382 ]
  %.lcssa493.ph = phi <4 x i64> [ undef, %vector.ph387 ], [ %438, %vector.body382 ]
  %.lcssa492.ph = phi <4 x i64> [ undef, %vector.ph387 ], [ %439, %vector.body382 ]
  %.lcssa491.ph = phi <4 x i64> [ undef, %vector.ph387 ], [ %440, %vector.body382 ]
  %index390.unr = phi i64 [ 0, %vector.ph387 ], [ %index.next391.3, %vector.body382 ]
  %vec.phi400.unr = phi <4 x i64> [ zeroinitializer, %vector.ph387 ], [ %437, %vector.body382 ]
  %vec.phi401.unr = phi <4 x i64> [ zeroinitializer, %vector.ph387 ], [ %438, %vector.body382 ]
  %vec.phi402.unr = phi <4 x i64> [ zeroinitializer, %vector.ph387 ], [ %439, %vector.body382 ]
  %vec.phi403.unr = phi <4 x i64> [ zeroinitializer, %vector.ph387 ], [ %440, %vector.body382 ]
  %lcmp.mod520 = icmp eq i64 %xtraiter518, 0
  br i1 %lcmp.mod520, label %middle.block383, label %vector.body382.epil.preheader

vector.body382.epil.preheader:                    ; preds = %middle.block383.unr-lcssa
  br label %vector.body382.epil

vector.body382.epil:                              ; preds = %vector.body382.epil, %vector.body382.epil.preheader
  %index390.epil = phi i64 [ %index390.unr, %vector.body382.epil.preheader ], [ %index.next391.epil, %vector.body382.epil ]
  %vec.phi400.epil = phi <4 x i64> [ %vec.phi400.unr, %vector.body382.epil.preheader ], [ %453, %vector.body382.epil ]
  %vec.phi401.epil = phi <4 x i64> [ %vec.phi401.unr, %vector.body382.epil.preheader ], [ %454, %vector.body382.epil ]
  %vec.phi402.epil = phi <4 x i64> [ %vec.phi402.unr, %vector.body382.epil.preheader ], [ %455, %vector.body382.epil ]
  %vec.phi403.epil = phi <4 x i64> [ %vec.phi403.unr, %vector.body382.epil.preheader ], [ %456, %vector.body382.epil ]
  %epil.iter519 = phi i64 [ %xtraiter518, %vector.body382.epil.preheader ], [ %epil.iter519.sub, %vector.body382.epil ]
  %441 = getelementptr inbounds i8, i8* %add.ptr.i190, i64 %index390.epil
  %442 = bitcast i8* %441 to <4 x i8>*
  %wide.load404.epil = load <4 x i8>, <4 x i8>* %442, align 1, !tbaa !7
  %443 = getelementptr i8, i8* %441, i64 4
  %444 = bitcast i8* %443 to <4 x i8>*
  %wide.load405.epil = load <4 x i8>, <4 x i8>* %444, align 1, !tbaa !7
  %445 = getelementptr i8, i8* %441, i64 8
  %446 = bitcast i8* %445 to <4 x i8>*
  %wide.load406.epil = load <4 x i8>, <4 x i8>* %446, align 1, !tbaa !7
  %447 = getelementptr i8, i8* %441, i64 12
  %448 = bitcast i8* %447 to <4 x i8>*
  %wide.load407.epil = load <4 x i8>, <4 x i8>* %448, align 1, !tbaa !7
  %449 = zext <4 x i8> %wide.load404.epil to <4 x i64>
  %450 = zext <4 x i8> %wide.load405.epil to <4 x i64>
  %451 = zext <4 x i8> %wide.load406.epil to <4 x i64>
  %452 = zext <4 x i8> %wide.load407.epil to <4 x i64>
  %453 = add nuw nsw <4 x i64> %vec.phi400.epil, %449
  %454 = add nuw nsw <4 x i64> %vec.phi401.epil, %450
  %455 = add nuw nsw <4 x i64> %vec.phi402.epil, %451
  %456 = add nuw nsw <4 x i64> %vec.phi403.epil, %452
  %index.next391.epil = add i64 %index390.epil, 16
  %epil.iter519.sub = add i64 %epil.iter519, -1
  %epil.iter519.cmp = icmp eq i64 %epil.iter519.sub, 0
  br i1 %epil.iter519.cmp, label %middle.block383, label %vector.body382.epil, !llvm.loop !104

middle.block383:                                  ; preds = %vector.body382.epil, %middle.block383.unr-lcssa
  %.lcssa494 = phi <4 x i64> [ %.lcssa494.ph, %middle.block383.unr-lcssa ], [ %453, %vector.body382.epil ]
  %.lcssa493 = phi <4 x i64> [ %.lcssa493.ph, %middle.block383.unr-lcssa ], [ %454, %vector.body382.epil ]
  %.lcssa492 = phi <4 x i64> [ %.lcssa492.ph, %middle.block383.unr-lcssa ], [ %455, %vector.body382.epil ]
  %.lcssa491 = phi <4 x i64> [ %.lcssa491.ph, %middle.block383.unr-lcssa ], [ %456, %vector.body382.epil ]
  %bin.rdx408 = add nuw <4 x i64> %.lcssa493, %.lcssa494
  %bin.rdx409 = add <4 x i64> %.lcssa492, %bin.rdx408
  %bin.rdx410 = add <4 x i64> %.lcssa491, %bin.rdx409
  %rdx.shuf411 = shufflevector <4 x i64> %bin.rdx410, <4 x i64> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx412 = add <4 x i64> %bin.rdx410, %rdx.shuf411
  %rdx.shuf413 = shufflevector <4 x i64> %bin.rdx412, <4 x i64> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx414 = add <4 x i64> %bin.rdx412, %rdx.shuf413
  %457 = extractelement <4 x i64> %bin.rdx414, i32 0
  %cmp.n393 = icmp eq i64 %324, %n.vec389
  br i1 %cmp.n393, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i239, label %for.body29.i.i235.preheader

for.body29.i.i235.preheader:                      ; preds = %middle.block383, %for.body29.lr.ph.i.i228
  %j25.068.i.i229.ph = phi i64 [ 0, %for.body29.lr.ph.i.i228 ], [ %n.vec389, %middle.block383 ]
  %r.167.i.i230.ph = phi i64 [ 0, %for.body29.lr.ph.i.i228 ], [ %457, %middle.block383 ]
  br label %for.body29.i.i235

for.body29.i.i235:                                ; preds = %for.body29.i.i235.preheader, %for.body29.i.i235
  %j25.068.i.i229 = phi i64 [ %inc34.i.i233, %for.body29.i.i235 ], [ %j25.068.i.i229.ph, %for.body29.i.i235.preheader ]
  %r.167.i.i230 = phi i64 [ %add32.i.i232, %for.body29.i.i235 ], [ %r.167.i.i230.ph, %for.body29.i.i235.preheader ]
  %arrayidx30.i.i231 = getelementptr inbounds i8, i8* %add.ptr.i190, i64 %j25.068.i.i229
  %458 = load i8, i8* %arrayidx30.i.i231, align 1, !tbaa !7, !range !73
  %459 = zext i8 %458 to i64
  %add32.i.i232 = add nuw nsw i64 %r.167.i.i230, %459
  %inc34.i.i233 = add nuw nsw i64 %j25.068.i.i229, 1
  %exitcond73.i.i234 = icmp eq i64 %inc34.i.i233, %sub.i191
  br i1 %exitcond73.i.i234, label %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i239, label %for.body29.i.i235, !llvm.loop !105

_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i239: ; preds = %for.body29.i.i235, %vector.ph353, %middle.block383, %if.else.i.i227, %if.then.i.i202
  %r.2.i.i236 = phi i64 [ 0, %if.then.i.i202 ], [ 0, %if.else.i.i227 ], [ %457, %middle.block383 ], [ %add21.i.i214, %vector.ph353 ], [ %add32.i.i232, %for.body29.i.i235 ]
  %mul.i237 = shl i64 %r.2.i.i236, 3
  %call1.i238 = tail call noalias i8* @malloc(i64 %mul.i237) #2
  %460 = bitcast i8* %call1.i238 to i64*
  br label %if.end.i242

if.end.i242:                                      ; preds = %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i239, %pfor.body69
  %Out.addr.0.i240 = phi i64* [ %460, %_ZN8sequence14sumFlagsSerialIlEET_PbS1_.exit.i239 ], [ %add.ptr83, %pfor.body69 ]
  %cmp220.i241 = icmp sgt i64 %.sroa.speculated275, %add72
  br i1 %cmp220.i241, label %for.body.i251.preheader, label %invoke.cont86

for.body.i251.preheader:                          ; preds = %if.end.i242
  %xtraiter528 = and i64 %309, 3
  %461 = icmp ult i64 %311, 3
  br i1 %461, label %invoke.cont86.loopexit.unr-lcssa, label %for.body.i251.preheader.new

for.body.i251.preheader.new:                      ; preds = %for.body.i251.preheader
  %unroll_iter531 = sub i64 %309, %xtraiter528
  br label %for.body.i251

for.body.i251:                                    ; preds = %for.inc.i258.3, %for.body.i251.preheader.new
  %k.022.i247 = phi i64 [ 0, %for.body.i251.preheader.new ], [ %k.1.i255.3, %for.inc.i258.3 ]
  %storemerge21.i248 = phi i64 [ %add72, %for.body.i251.preheader.new ], [ %inc7.i256.3, %for.inc.i258.3 ]
  %niter532 = phi i64 [ %unroll_iter531, %for.body.i251.preheader.new ], [ %niter532.nsub.3, %for.inc.i258.3 ]
  %arrayidx.i249 = getelementptr inbounds i8, i8* %Fl, i64 %storemerge21.i248
  %462 = load i8, i8* %arrayidx.i249, align 1, !tbaa !7, !range !73
  %tobool.i250 = icmp eq i8 %462, 0
  br i1 %tobool.i250, label %for.inc.i258, label %if.then3.i254

if.then3.i254:                                    ; preds = %for.body.i251
  %inc.i252 = add nsw i64 %k.022.i247, 1
  %arrayidx5.i253 = getelementptr inbounds i64, i64* %Out.addr.0.i240, i64 %k.022.i247
  store i64 %storemerge21.i248, i64* %arrayidx5.i253, align 8, !tbaa !10
  br label %for.inc.i258

for.inc.i258:                                     ; preds = %if.then3.i254, %for.body.i251
  %k.1.i255 = phi i64 [ %inc.i252, %if.then3.i254 ], [ %k.022.i247, %for.body.i251 ]
  %inc7.i256 = add nsw i64 %storemerge21.i248, 1
  %arrayidx.i249.1 = getelementptr inbounds i8, i8* %Fl, i64 %inc7.i256
  %463 = load i8, i8* %arrayidx.i249.1, align 1, !tbaa !7, !range !73
  %tobool.i250.1 = icmp eq i8 %463, 0
  br i1 %tobool.i250.1, label %for.inc.i258.1, label %if.then3.i254.1

invoke.cont86.loopexit.unr-lcssa:                 ; preds = %for.inc.i258.3, %for.body.i251.preheader
  %k.022.i247.unr = phi i64 [ 0, %for.body.i251.preheader ], [ %k.1.i255.3, %for.inc.i258.3 ]
  %storemerge21.i248.unr = phi i64 [ %add72, %for.body.i251.preheader ], [ %inc7.i256.3, %for.inc.i258.3 ]
  %lcmp.mod530 = icmp eq i64 %xtraiter528, 0
  br i1 %lcmp.mod530, label %invoke.cont86, label %for.body.i251.epil.preheader

for.body.i251.epil.preheader:                     ; preds = %invoke.cont86.loopexit.unr-lcssa
  br label %for.body.i251.epil

for.body.i251.epil:                               ; preds = %for.inc.i258.epil, %for.body.i251.epil.preheader
  %k.022.i247.epil = phi i64 [ %k.1.i255.epil, %for.inc.i258.epil ], [ %k.022.i247.unr, %for.body.i251.epil.preheader ]
  %storemerge21.i248.epil = phi i64 [ %inc7.i256.epil, %for.inc.i258.epil ], [ %storemerge21.i248.unr, %for.body.i251.epil.preheader ]
  %epil.iter529 = phi i64 [ %epil.iter529.sub, %for.inc.i258.epil ], [ %xtraiter528, %for.body.i251.epil.preheader ]
  %arrayidx.i249.epil = getelementptr inbounds i8, i8* %Fl, i64 %storemerge21.i248.epil
  %464 = load i8, i8* %arrayidx.i249.epil, align 1, !tbaa !7, !range !73
  %tobool.i250.epil = icmp eq i8 %464, 0
  br i1 %tobool.i250.epil, label %for.inc.i258.epil, label %if.then3.i254.epil

if.then3.i254.epil:                               ; preds = %for.body.i251.epil
  %inc.i252.epil = add nsw i64 %k.022.i247.epil, 1
  %arrayidx5.i253.epil = getelementptr inbounds i64, i64* %Out.addr.0.i240, i64 %k.022.i247.epil
  store i64 %storemerge21.i248.epil, i64* %arrayidx5.i253.epil, align 8, !tbaa !10
  br label %for.inc.i258.epil

for.inc.i258.epil:                                ; preds = %if.then3.i254.epil, %for.body.i251.epil
  %k.1.i255.epil = phi i64 [ %inc.i252.epil, %if.then3.i254.epil ], [ %k.022.i247.epil, %for.body.i251.epil ]
  %inc7.i256.epil = add nsw i64 %storemerge21.i248.epil, 1
  %epil.iter529.sub = add i64 %epil.iter529, -1
  %epil.iter529.cmp = icmp eq i64 %epil.iter529.sub, 0
  br i1 %epil.iter529.cmp, label %invoke.cont86, label %for.body.i251.epil, !llvm.loop !106

invoke.cont86:                                    ; preds = %invoke.cont86.loopexit.unr-lcssa, %for.inc.i258.epil, %if.end.i242
  reattach within %syncreg53, label %pfor.inc92

pfor.inc92:                                       ; preds = %pfor.detach64, %invoke.cont86
  %inc93 = add nuw nsw i64 %__begin55.0288, 1
  %exitcond = icmp eq i64 %__begin55.0288, %smax
  br i1 %exitcond, label %pfor.cond.cleanup63, label %pfor.detach64, !llvm.loop !107

lpad94:                                           ; preds = %pfor.detach64
  %465 = landingpad { i8*, i32 }
          cleanup
  %466 = extractvalue { i8*, i32 } %465, 0
  %467 = extractvalue { i8*, i32 } %465, 1
  sync within %syncreg53, label %ehcleanup113

sync.continue101:                                 ; preds = %pfor.cond.cleanup63
  tail call void @free(i8* %call2) #2
  br label %cleanup

ehcleanup113:                                     ; preds = %lpad94, %lpad26
  %ehselector.slot28.0 = phi i32 [ %298, %lpad26 ], [ %467, %lpad94 ]
  %exn.slot27.0 = phi i8* [ %297, %lpad26 ], [ %466, %lpad94 ]
  %lpad.val118 = insertvalue { i8*, i32 } undef, i8* %exn.slot27.0, 0
  %lpad.val119 = insertvalue { i8*, i32 } %lpad.val118, i32 %ehselector.slot28.0, 1
  resume { i8*, i32 } %lpad.val119

cleanup.loopexit.unr-lcssa:                       ; preds = %for.inc.i.3, %for.body.i.preheader
  %k.1.i.lcssa.ph = phi i64 [ undef, %for.body.i.preheader ], [ %k.1.i.3, %for.inc.i.3 ]
  %k.022.i.unr = phi i64 [ 0, %for.body.i.preheader ], [ %k.1.i.3, %for.inc.i.3 ]
  %storemerge21.i.unr = phi i64 [ %s, %for.body.i.preheader ], [ %inc7.i.3, %for.inc.i.3 ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %cleanup, label %for.body.i.epil.preheader

for.body.i.epil.preheader:                        ; preds = %cleanup.loopexit.unr-lcssa
  br label %for.body.i.epil

for.body.i.epil:                                  ; preds = %for.inc.i.epil, %for.body.i.epil.preheader
  %k.022.i.epil = phi i64 [ %k.1.i.epil, %for.inc.i.epil ], [ %k.022.i.unr, %for.body.i.epil.preheader ]
  %storemerge21.i.epil = phi i64 [ %inc7.i.epil, %for.inc.i.epil ], [ %storemerge21.i.unr, %for.body.i.epil.preheader ]
  %epil.iter = phi i64 [ %epil.iter.sub, %for.inc.i.epil ], [ %xtraiter, %for.body.i.epil.preheader ]
  %arrayidx.i.epil = getelementptr inbounds i8, i8* %Fl, i64 %storemerge21.i.epil
  %468 = load i8, i8* %arrayidx.i.epil, align 1, !tbaa !7, !range !73
  %tobool.i.epil = icmp eq i8 %468, 0
  br i1 %tobool.i.epil, label %for.inc.i.epil, label %if.then3.i.epil

if.then3.i.epil:                                  ; preds = %for.body.i.epil
  %inc.i.epil = add nsw i64 %k.022.i.epil, 1
  %arrayidx5.i.epil = getelementptr inbounds i64, i64* %Out.addr.0.i, i64 %k.022.i.epil
  store i64 %storemerge21.i.epil, i64* %arrayidx5.i.epil, align 8, !tbaa !10
  br label %for.inc.i.epil

for.inc.i.epil:                                   ; preds = %if.then3.i.epil, %for.body.i.epil
  %k.1.i.epil = phi i64 [ %inc.i.epil, %if.then3.i.epil ], [ %k.022.i.epil, %for.body.i.epil ]
  %inc7.i.epil = add nsw i64 %storemerge21.i.epil, 1
  %epil.iter.sub = add i64 %epil.iter, -1
  %epil.iter.cmp = icmp eq i64 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %cleanup, label %for.body.i.epil, !llvm.loop !108

cleanup:                                          ; preds = %cleanup.loopexit.unr-lcssa, %for.inc.i.epil, %if.end.i, %sync.continue101
  %retval.sroa.0.0 = phi i64* [ %Out.addr.0, %sync.continue101 ], [ %Out.addr.0.i, %if.end.i ], [ %Out.addr.0.i, %for.inc.i.epil ], [ %Out.addr.0.i, %cleanup.loopexit.unr-lcssa ]
  %retval.sroa.3.0 = phi i64 [ %call.i, %sync.continue101 ], [ 0, %if.end.i ], [ %k.1.i.lcssa.ph, %cleanup.loopexit.unr-lcssa ], [ %k.1.i.epil, %for.inc.i.epil ]
  %.fca.0.insert = insertvalue { i64*, i64 } undef, i64* %retval.sroa.0.0, 0
  %.fca.1.insert = insertvalue { i64*, i64 } %.fca.0.insert, i64 %retval.sroa.3.0, 1
  ret { i64*, i64 } %.fca.1.insert

if.then3.i.1:                                     ; preds = %for.inc.i
  %inc.i.1 = add nsw i64 %k.1.i, 1
  %arrayidx5.i.1 = getelementptr inbounds i64, i64* %Out.addr.0.i, i64 %k.1.i
  store i64 %inc7.i, i64* %arrayidx5.i.1, align 8, !tbaa !10
  br label %for.inc.i.1

for.inc.i.1:                                      ; preds = %if.then3.i.1, %for.inc.i
  %k.1.i.1 = phi i64 [ %inc.i.1, %if.then3.i.1 ], [ %k.1.i, %for.inc.i ]
  %inc7.i.1 = add nsw i64 %storemerge21.i, 2
  %arrayidx.i.2 = getelementptr inbounds i8, i8* %Fl, i64 %inc7.i.1
  %469 = load i8, i8* %arrayidx.i.2, align 1, !tbaa !7, !range !73
  %tobool.i.2 = icmp eq i8 %469, 0
  br i1 %tobool.i.2, label %for.inc.i.2, label %if.then3.i.2

if.then3.i.2:                                     ; preds = %for.inc.i.1
  %inc.i.2 = add nsw i64 %k.1.i.1, 1
  %arrayidx5.i.2 = getelementptr inbounds i64, i64* %Out.addr.0.i, i64 %k.1.i.1
  store i64 %inc7.i.1, i64* %arrayidx5.i.2, align 8, !tbaa !10
  br label %for.inc.i.2

for.inc.i.2:                                      ; preds = %if.then3.i.2, %for.inc.i.1
  %k.1.i.2 = phi i64 [ %inc.i.2, %if.then3.i.2 ], [ %k.1.i.1, %for.inc.i.1 ]
  %inc7.i.2 = add nsw i64 %storemerge21.i, 3
  %arrayidx.i.3 = getelementptr inbounds i8, i8* %Fl, i64 %inc7.i.2
  %470 = load i8, i8* %arrayidx.i.3, align 1, !tbaa !7, !range !73
  %tobool.i.3 = icmp eq i8 %470, 0
  br i1 %tobool.i.3, label %for.inc.i.3, label %if.then3.i.3

if.then3.i.3:                                     ; preds = %for.inc.i.2
  %inc.i.3 = add nsw i64 %k.1.i.2, 1
  %arrayidx5.i.3 = getelementptr inbounds i64, i64* %Out.addr.0.i, i64 %k.1.i.2
  store i64 %inc7.i.2, i64* %arrayidx5.i.3, align 8, !tbaa !10
  br label %for.inc.i.3

for.inc.i.3:                                      ; preds = %if.then3.i.3, %for.inc.i.2
  %k.1.i.3 = phi i64 [ %inc.i.3, %if.then3.i.3 ], [ %k.1.i.2, %for.inc.i.2 ]
  %inc7.i.3 = add nsw i64 %storemerge21.i, 4
  %niter.nsub.3 = add i64 %niter, -4
  %niter.ncmp.3 = icmp eq i64 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %cleanup.loopexit.unr-lcssa, label %for.body.i

if.then3.i254.1:                                  ; preds = %for.inc.i258
  %inc.i252.1 = add nsw i64 %k.1.i255, 1
  %arrayidx5.i253.1 = getelementptr inbounds i64, i64* %Out.addr.0.i240, i64 %k.1.i255
  store i64 %inc7.i256, i64* %arrayidx5.i253.1, align 8, !tbaa !10
  br label %for.inc.i258.1

for.inc.i258.1:                                   ; preds = %if.then3.i254.1, %for.inc.i258
  %k.1.i255.1 = phi i64 [ %inc.i252.1, %if.then3.i254.1 ], [ %k.1.i255, %for.inc.i258 ]
  %inc7.i256.1 = add nsw i64 %storemerge21.i248, 2
  %arrayidx.i249.2 = getelementptr inbounds i8, i8* %Fl, i64 %inc7.i256.1
  %471 = load i8, i8* %arrayidx.i249.2, align 1, !tbaa !7, !range !73
  %tobool.i250.2 = icmp eq i8 %471, 0
  br i1 %tobool.i250.2, label %for.inc.i258.2, label %if.then3.i254.2

if.then3.i254.2:                                  ; preds = %for.inc.i258.1
  %inc.i252.2 = add nsw i64 %k.1.i255.1, 1
  %arrayidx5.i253.2 = getelementptr inbounds i64, i64* %Out.addr.0.i240, i64 %k.1.i255.1
  store i64 %inc7.i256.1, i64* %arrayidx5.i253.2, align 8, !tbaa !10
  br label %for.inc.i258.2

for.inc.i258.2:                                   ; preds = %if.then3.i254.2, %for.inc.i258.1
  %k.1.i255.2 = phi i64 [ %inc.i252.2, %if.then3.i254.2 ], [ %k.1.i255.1, %for.inc.i258.1 ]
  %inc7.i256.2 = add nsw i64 %storemerge21.i248, 3
  %arrayidx.i249.3 = getelementptr inbounds i8, i8* %Fl, i64 %inc7.i256.2
  %472 = load i8, i8* %arrayidx.i249.3, align 1, !tbaa !7, !range !73
  %tobool.i250.3 = icmp eq i8 %472, 0
  br i1 %tobool.i250.3, label %for.inc.i258.3, label %if.then3.i254.3

if.then3.i254.3:                                  ; preds = %for.inc.i258.2
  %inc.i252.3 = add nsw i64 %k.1.i255.2, 1
  %arrayidx5.i253.3 = getelementptr inbounds i64, i64* %Out.addr.0.i240, i64 %k.1.i255.2
  store i64 %inc7.i256.2, i64* %arrayidx5.i253.3, align 8, !tbaa !10
  br label %for.inc.i258.3

for.inc.i258.3:                                   ; preds = %if.then3.i254.3, %for.inc.i258.2
  %k.1.i255.3 = phi i64 [ %inc.i252.3, %if.then3.i254.3 ], [ %k.1.i255.2, %for.inc.i258.2 ]
  %inc7.i256.3 = add nsw i64 %storemerge21.i248, 4
  %niter532.nsub.3 = add i64 %niter532, -4
  %niter532.ncmp.3 = icmp eq i64 %niter532.nsub.3, 0
  br i1 %niter532.ncmp.3, label %invoke.cont86.loopexit.unr-lcssa, label %for.body.i251
}

; CHECK-LABEL: define private fastcc void @_ZN8sequence4packIllN5utils9identityFIlEEEE4_seqIT_EPS5_PbT0_S9_T1_.outline_pfor.detach64.ls1(i64 %__begin55.0288.start.ls1,
; CHECK: %__begin55.0288.ls1.dac = phi i64
; CHECK: [ %__begin55.0288.start.ls1, %pfor.detach64.lr.ph.ls1 ]
; CHECK: [ %[[NEXTITERVAR:.+]], %[[RECURCONT:.+]] ]
; CHECK: [[RECURCONT]]:
; CHECK: %[[NEXTITERVAR]] = add nuw nsw i64 {{.+}}, 1

; CHECK-LABEL: define private fastcc void @_ZN8sequence4packIllN5utils9identityFIlEEEE4_seqIT_EPS5_PbT0_S9_T1_.outline_pfor.detach.ls1(i64 %__begin.0290.start.ls1
; CHECK: %__begin.0290.ls1.dac = phi i64
; CHECK: [ %__begin.0290.start.ls1, %pfor.detach.lr.ph.ls1 ]
; CHECK: [ %[[NEXTITERVAR:.+]], %[[RECURCONT:.+]] ]
; CHECK: [[RECURCONT]]:
; CHECK: %[[NEXTITERVAR]] = add nuw nsw i64 {{.+}}, 1

declare i64 @_ZN8sequence4scanIllN5utils4addFIlEENS_4getAIllEEEET_PS6_T0_S8_T1_T2_S6_bb(i64* %Out, i64 %s, i64 %e, i64* %g.coerce, i64 %zero, i1 zeroext %inclusive, i1 zeroext %back) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #1

!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!6 = !{!"tapir.loop.spawn.strategy", i32 1}
!7 = !{!8, !8, i64 0}
!8 = !{!"bool", !3, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"long", !3, i64 0}
!27 = !{!"int", !3, i64 0}
!55 = !{!27, !27, i64 0}
!73 = !{i8 0, i8 2}
!76 = !{!"llvm.loop.isvectorized", i32 1}
!78 = !{!"llvm.loop.unroll.runtime.disable"}
!82 = !{!"llvm.loop.unroll.disable"}
!96 = distinct !{!96, !76}
!97 = distinct !{!97, !82}
!98 = distinct !{!98, !78, !76}
!99 = distinct !{!99, !76}
!100 = distinct !{!100, !82}
!101 = distinct !{!101, !78, !76}
!102 = distinct !{!102, !6}
!103 = distinct !{!103, !76}
!104 = distinct !{!104, !82}
!105 = distinct !{!105, !78, !76}
!106 = distinct !{!106, !82}
!107 = distinct !{!107, !6}
!108 = distinct !{!108, !82}
