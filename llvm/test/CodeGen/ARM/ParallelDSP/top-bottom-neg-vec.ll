; RUN: opt -mtriple=thumbv7-unknown-linux-android -arm-parallel-dsp -S %s -o - | FileCheck %s

@a = local_unnamed_addr global i32 0, align 4
@b = local_unnamed_addr global i8* null, align 4
@c = local_unnamed_addr global i8 0, align 1
@d = local_unnamed_addr global i16* null, align 4

; CHECK-LABEL: @convolve
; CHECK-NOT: bitcast i16* [[ANY:%[^ ]+]] to i32*
define void @convolve() local_unnamed_addr #0 {
entry:
  br label %for.cond

for.cond:
  %e.0 = phi i32 [ undef, %entry ], [ %e.1.lcssa, %for.end ]
  %f.0 = phi i32 [ undef, %entry ], [ %f.1.lcssa, %for.end ]
  %g.0 = phi i32 [ undef, %entry ], [ %g.1.lcssa, %for.end ]
  %cmp13 = icmp slt i32 %g.0, 1
  br i1 %cmp13, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = load i16*, i16** @d, align 4
  %1 = load i8*, i8** @b, align 4
  %2 = load i32, i32* @a, align 4
  %3 = sub i32 1, %g.0
  %min.iters.check = icmp ugt i32 %3, 3
  %ident.check = icmp eq i32 %2, 1
  %or.cond = and i1 %min.iters.check, %ident.check
  br i1 %or.cond, label %vector.ph, label %for.body.preheader

vector.ph:
  %n.vec = and i32 %3, -4
  %ind.end = add i32 %g.0, %n.vec
  %4 = mul i32 %2, %n.vec
  %ind.end20 = add i32 %f.0, %4
  %5 = insertelement <4 x i32> <i32 undef, i32 0, i32 0, i32 0>, i32 %e.0, i32 0
  br label %vector.body

vector.body:
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ %5, %vector.ph ], [ %14, %vector.body ]
  %offset.idx = add i32 %g.0, %index
  %6 = mul i32 %2, %index
  %offset.idx21 = add i32 %f.0, %6
  %7 = getelementptr inbounds i16, i16* %0, i32 %offset.idx
  %8 = bitcast i16* %7 to <4 x i16>*
  %wide.load = load <4 x i16>, <4 x i16>* %8, align 2
  %9 = sext <4 x i16> %wide.load to <4 x i32>
  %10 = getelementptr inbounds i8, i8* %1, i32 %offset.idx21
  %11 = bitcast i8* %10 to <4 x i8>*
  %wide.load25 = load <4 x i8>, <4 x i8>* %11, align 1
  %12 = zext <4 x i8> %wide.load25 to <4 x i32>
  %13 = mul nsw <4 x i32> %12, %9
  %14 = add nsw <4 x i32> %13, %vec.phi
  %index.next = add i32 %index, 4
  %15 = icmp eq i32 %index.next, %n.vec
  br i1 %15, label %middle.block, label %vector.body

middle.block:
  %rdx.shuf = shufflevector <4 x i32> %14, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = add <4 x i32> %14, %rdx.shuf
  %rdx.shuf26 = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx27 = add <4 x i32> %bin.rdx, %rdx.shuf26
  %16 = extractelement <4 x i32> %bin.rdx27, i32 0
  %cmp.n = icmp eq i32 %3, %n.vec
  br i1 %cmp.n, label %for.end, label %for.body.preheader

for.body.preheader:
  %g.116.ph = phi i32 [ %g.0, %for.body.lr.ph ], [ %ind.end, %middle.block ]
  %f.115.ph = phi i32 [ %f.0, %for.body.lr.ph ], [ %ind.end20, %middle.block ]
  %e.114.ph = phi i32 [ %e.0, %for.body.lr.ph ], [ %16, %middle.block ]
  br label %for.body

for.body:
  %g.116 = phi i32 [ %inc, %for.body ], [ %g.116.ph, %for.body.preheader ]
  %f.115 = phi i32 [ %add4, %for.body ], [ %f.115.ph, %for.body.preheader ]
  %e.114 = phi i32 [ %add, %for.body ], [ %e.114.ph, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %0, i32 %g.116
  %17 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %17 to i32
  %arrayidx2 = getelementptr inbounds i8, i8* %1, i32 %f.115
  %18 = load i8, i8* %arrayidx2, align 1
  %conv3 = zext i8 %18 to i32
  %mul = mul nsw i32 %conv3, %conv
  %add = add nsw i32 %mul, %e.114
  %inc = add nsw i32 %g.116, 1
  %add4 = add nsw i32 %2, %f.115
  %cmp = icmp slt i32 %g.116, 0
  br i1 %cmp, label %for.body, label %for.end

for.end:
  %e.1.lcssa = phi i32 [ %e.0, %for.cond ], [ %16, %middle.block ], [ %add, %for.body ]
  %f.1.lcssa = phi i32 [ %f.0, %for.cond ], [ %ind.end20, %middle.block ], [ %add4, %for.body ]
  %g.1.lcssa = phi i32 [ %g.0, %for.cond ], [ %ind.end, %middle.block ], [ %inc, %for.body ]
  %conv5 = trunc i32 %e.1.lcssa to i8
  store i8 %conv5, i8* @c, align 1
  br label %for.cond
}
