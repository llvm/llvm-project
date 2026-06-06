; During lowering of HVX instruction for 64B vector, the rotation
; direction for VROR (as part of concat of vectors lowering) is fixed.

; RUN: llc -mtriple=hexagon -O2 %s -o - | FileCheck %s

; CHECK: vec.epilog.ph
; CHECK: r{{.*}} = {{.*}}#48
; CHECK: vec.epilog.vector.body


%struct.str = type { i8, i8, i8 }

define dso_local void @foo(i16* nocapture noundef writeonly %pOut, i16* nocapture noundef readonly %Coefs, %struct.str* nocapture noundef readonly %pQ, i32 noundef %Intra) local_unnamed_addr #0 {
entry:
  %Coefs13 = ptrtoint i16* %Coefs to i32
  %pOut12 = ptrtoint i16* %pOut to i32
  %cmp10 = icmp slt i32 %Intra, 16
  br i1 %cmp10, label %iter.check, label %for.end

iter.check:                                       ; preds = %entry
  %Q = getelementptr inbounds %struct.str, %struct.str* %pQ, i32 0, i32 0
  %0 = load i8, i8* %Q, align 1
  %conv3 = zext i8 %0 to i32
  %1 = sub nsw i32 0, %conv3
  %2 = sub i32 16, %Intra
  %min.iters.check = icmp ult i32 %2, 8
  br i1 %min.iters.check, label %for.body.preheader, label %vector.memcheck

vector.memcheck:                                  ; preds = %iter.check
  %3 = shl i32 %Intra, 1
  %4 = add i32 %3, %pOut12
  %5 = add i32 %3, %Coefs13
  %6 = sub i32 %4, %5
  %diff.check = icmp ult i32 %6, 128
  br i1 %diff.check, label %for.body.preheader, label %vector.main.loop.iter.check

vector.main.loop.iter.check:                      ; preds = %vector.memcheck
  %min.iters.check14 = icmp ult i32 %2, 64
  br i1 %min.iters.check14, label %vec.epilog.ph, label %vector.ph

vector.ph:                                        ; preds = %vector.main.loop.iter.check
  %n.vec = and i32 %2, -64
  %bd.spinsert = insertelement <32 x i32> poison, i32 %1, i64 0
  %bd.sp = shufflevector <32 x i32> %bd.spinsert, <32 x i32> poison, <32 x i32> zeroinitializer
  %bd.spinsert16 = insertelement <32 x i32> poison, i32 %conv3, i64 0
  %bd.sp17 = shufflevector <32 x i32> %bd.spinsert16, <32 x i32> poison, <32 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %offset.idx = add i32 %index, %Intra
  %7 = getelementptr inbounds i16, i16* %Coefs, i32 %offset.idx
  %8 = bitcast i16* %7 to <32 x i16>*
  %wide.load = load <32 x i16>, <32 x i16>* %8, align 2
  %9 = getelementptr inbounds i16, i16* %7, i32 32
  %10 = bitcast i16* %9 to <32 x i16>*
  %wide.load15 = load <32 x i16>, <32 x i16>* %10, align 2
  %11 = icmp slt <32 x i16> %wide.load, zeroinitializer
  %12 = icmp slt <32 x i16> %wide.load15, zeroinitializer
  %13 = select <32 x i1> %11, <32 x i32> %bd.sp, <32 x i32> %bd.sp17
  %14 = select <32 x i1> %12, <32 x i32> %bd.sp, <32 x i32> %bd.sp17
  %15 = trunc <32 x i32> %13 to <32 x i16>
  %16 = trunc <32 x i32> %14 to <32 x i16>
  %17 = getelementptr inbounds i16, i16* %pOut, i32 %offset.idx
  %18 = bitcast i16* %17 to <32 x i16>*
  store <32 x i16> %15, <32 x i16>* %18, align 2
  %19 = getelementptr inbounds i16, i16* %17, i32 32
  %20 = bitcast i16* %19 to <32 x i16>*
  store <32 x i16> %16, <32 x i16>* %20, align 2
  %index.next = add nuw i32 %index, 64
  %21 = icmp eq i32 %index.next, %n.vec
  br i1 %21, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i32 %2, %n.vec
  br i1 %cmp.n, label %for.end, label %vec.epilog.iter.check

vec.epilog.iter.check:                            ; preds = %middle.block
  %ind.end24 = add i32 %n.vec, %Intra
  %n.vec.remaining = and i32 %2, 56
  %min.epilog.iters.check = icmp eq i32 %n.vec.remaining, 0
  br i1 %min.epilog.iters.check, label %for.body.preheader, label %vec.epilog.ph

vec.epilog.ph:                                    ; preds = %vector.main.loop.iter.check, %vec.epilog.iter.check
  %vec.epilog.resume.val = phi i32 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %n.vec23 = and i32 %2, -8
  %ind.end = add i32 %n.vec23, %Intra
  %bd.spinsert29 = insertelement <8 x i32> poison, i32 %1, i64 0
  %bd.sp30 = shufflevector <8 x i32> %bd.spinsert29, <8 x i32> poison, <8 x i32> zeroinitializer
  %bd.spinsert31 = insertelement <8 x i32> poison, i32 %conv3, i64 0
  %bd.sp32 = shufflevector <8 x i32> %bd.spinsert31, <8 x i32> poison, <8 x i32> zeroinitializer
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index26 = phi i32 [ %vec.epilog.resume.val, %vec.epilog.ph ], [ %index.next33, %vec.epilog.vector.body ]
  %offset.idx27 = add i32 %index26, %Intra
  %22 = getelementptr inbounds i16, i16* %Coefs, i32 %offset.idx27
  %23 = bitcast i16* %22 to <8 x i16>*
  %wide.load28 = load <8 x i16>, <8 x i16>* %23, align 2
  %24 = icmp slt <8 x i16> %wide.load28, zeroinitializer
  %25 = select <8 x i1> %24, <8 x i32> %bd.sp30, <8 x i32> %bd.sp32
  %26 = trunc <8 x i32> %25 to <8 x i16>
  %27 = getelementptr inbounds i16, i16* %pOut, i32 %offset.idx27
  %28 = bitcast i16* %27 to <8 x i16>*
  store <8 x i16> %26, <8 x i16>* %28, align 2
  %index.next33 = add nuw i32 %index26, 8
  %29 = icmp eq i32 %index.next33, %n.vec23
  br i1 %29, label %vec.epilog.middle.block, label %vec.epilog.vector.body

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  %cmp.n25 = icmp eq i32 %2, %n.vec23
  br i1 %cmp.n25, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %vector.memcheck, %iter.check, %vec.epilog.iter.check, %vec.epilog.middle.block
  %i.011.ph = phi i32 [ %Intra, %iter.check ], [ %Intra, %vector.memcheck ], [ %ind.end24, %vec.epilog.iter.check ], [ %ind.end, %vec.epilog.middle.block ]
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.011 = phi i32 [ %inc, %for.body ], [ %i.011.ph, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %Coefs, i32 %i.011
  %30 = load i16, i16* %arrayidx, align 2
  %cmp1 = icmp slt i16 %30, 0
  %31 = select i1 %cmp1, i32 %1, i32 %conv3
  %conv4 = trunc i32 %31 to i16
  %arrayidx5 = getelementptr inbounds i16, i16* %pOut, i32 %i.011
  store i16 %conv4, i16* %arrayidx5, align 2
  %inc = add i32 %i.011, 1
  %exitcond.not = icmp eq i32 %inc, 16
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %middle.block, %vec.epilog.middle.block, %entry
  ret void
}

attributes #0 = { argmemonly nofree norecurse nosync nounwind "target-cpu"="hexagonv66" "target-features"="+hvx-length64b,+hvxv66,+v66" }
