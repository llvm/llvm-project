; RUN: opt < %s -aa-pipeline=basic-aa -passes='require<da>,print<delinearization>' -disable-output
;
; a, b, c, d, g, h;
; char *f;
; static fn1(p1) {
;   char *e = p1;
;   for (; d;) {
;     a = 0;
;     for (;; ++a)
;       for (; b; ++b)
;         c = e[b + a];
;   }
; }
;
; fn2() {
;   for (;;)
;     fn1(&f[g * h]);
; }

@g = common global i32 0, align 4
@h = common global i32 0, align 4
@f = common global i8* null, align 4
@a = common global i32 0, align 4
@b = common global i32 0, align 4
@c = common global i32 0, align 4
@d = common global i32 0, align 4

define i32 @fn2() {
entry:
  %.pr = load i32, i32* @d, align 4
  %phitmp = icmp eq i32 %.pr, 0
  br label %for.cond

for.cond:
  %0 = phi i1 [ true, %for.cond ], [ %phitmp, %entry ]
  br i1 %0, label %for.cond, label %for.cond2thread-pre-split.preheader.i

for.cond2thread-pre-split.preheader.i:
  %1 = load i32, i32* @g, align 4
  %2 = load i32, i32* @h, align 4
  %mul = mul nsw i32 %2, %1
  %3 = load i8*, i8** @f, align 4
  %.pr.pre.i = load i32, i32* @b, align 4
  br label %for.cond2thread-pre-split.i

for.cond2thread-pre-split.i:
  %.pr.i = phi i32 [ 0, %for.inc5.i ], [ %.pr.pre.i, %for.cond2thread-pre-split.preheader.i ]
  %storemerge.i = phi i32 [ %inc6.i, %for.inc5.i ], [ 0, %for.cond2thread-pre-split.preheader.i ]
  store i32 %storemerge.i, i32* @a, align 4
  %tobool31.i = icmp eq i32 %.pr.i, 0
  br i1 %tobool31.i, label %for.inc5.i, label %for.body4.preheader.i

for.body4.preheader.i:
  %4 = icmp slt i32 %.pr.i, -7
  %add.i = add i32 %storemerge.i, %mul
  br i1 %4, label %for.body4.i.preheader, label %for.body4.ur.i.preheader

for.body4.i.preheader:
  %5 = sub i32 -8, %.pr.i
  %6 = lshr i32 %5, 3
  %7 = mul i32 %6, 8
  br label %for.body4.i

for.body4.i:
  %8 = phi i32 [ %inc.7.i, %for.body4.i ], [ %.pr.i, %for.body4.i.preheader ]
  %arrayidx.sum1 = add i32 %add.i, %8
  %arrayidx.i = getelementptr inbounds i8, i8* %3, i32 %arrayidx.sum1
  %9 = load i8, i8* %arrayidx.i, align 1
  %conv.i = sext i8 %9 to i32
  store i32 %conv.i, i32* @c, align 4
  %inc.i = add nsw i32 %8, 1
  store i32 %inc.i, i32* @b, align 4
  %arrayidx.sum2 = add i32 %add.i, %inc.i
  %arrayidx.1.i = getelementptr inbounds i8, i8* %3, i32 %arrayidx.sum2
  %10 = load i8, i8* %arrayidx.1.i, align 1
  %conv.1.i = sext i8 %10 to i32
  store i32 %conv.1.i, i32* @c, align 4
  %inc.1.i = add nsw i32 %8, 2
  store i32 %inc.1.i, i32* @b, align 4
  %arrayidx.sum3 = add i32 %add.i, %inc.1.i
  %arrayidx.2.i = getelementptr inbounds i8, i8* %3, i32 %arrayidx.sum3
  %11 = load i8, i8* %arrayidx.2.i, align 1
  %conv.2.i = sext i8 %11 to i32
  store i32 %conv.2.i, i32* @c, align 4
  %inc.2.i = add nsw i32 %8, 3
  store i32 %inc.2.i, i32* @b, align 4
  %arrayidx.sum4 = add i32 %add.i, %inc.2.i
  %arrayidx.3.i = getelementptr inbounds i8, i8* %3, i32 %arrayidx.sum4
  %12 = load i8, i8* %arrayidx.3.i, align 1
  %conv.3.i = sext i8 %12 to i32
  store i32 %conv.3.i, i32* @c, align 4
  %inc.3.i = add nsw i32 %8, 4
  store i32 %inc.3.i, i32* @b, align 4
  %arrayidx.sum5 = add i32 %add.i, %inc.3.i
  %arrayidx.4.i = getelementptr inbounds i8, i8* %3, i32 %arrayidx.sum5
  %13 = load i8, i8* %arrayidx.4.i, align 1
  %conv.4.i = sext i8 %13 to i32
  store i32 %conv.4.i, i32* @c, align 4
  %inc.4.i = add nsw i32 %8, 5
  store i32 %inc.4.i, i32* @b, align 4
  %arrayidx.sum6 = add i32 %add.i, %inc.4.i
  %arrayidx.5.i = getelementptr inbounds i8, i8* %3, i32 %arrayidx.sum6
  %14 = load i8, i8* %arrayidx.5.i, align 1
  %conv.5.i = sext i8 %14 to i32
  store i32 %conv.5.i, i32* @c, align 4
  %inc.5.i = add nsw i32 %8, 6
  store i32 %inc.5.i, i32* @b, align 4
  %arrayidx.sum7 = add i32 %add.i, %inc.5.i
  %arrayidx.6.i = getelementptr inbounds i8, i8* %3, i32 %arrayidx.sum7
  %15 = load i8, i8* %arrayidx.6.i, align 1
  %conv.6.i = sext i8 %15 to i32
  store i32 %conv.6.i, i32* @c, align 4
  %inc.6.i = add nsw i32 %8, 7
  store i32 %inc.6.i, i32* @b, align 4
  %arrayidx.sum8 = add i32 %add.i, %inc.6.i
  %arrayidx.7.i = getelementptr inbounds i8, i8* %3, i32 %arrayidx.sum8
  %16 = load i8, i8* %arrayidx.7.i, align 1
  %conv.7.i = sext i8 %16 to i32
  store i32 %conv.7.i, i32* @c, align 4
  %inc.7.i = add nsw i32 %8, 8
  store i32 %inc.7.i, i32* @b, align 4
  %tobool3.7.i = icmp sgt i32 %inc.7.i, -8
  br i1 %tobool3.7.i, label %for.inc5.loopexit.ur-lcssa.i, label %for.body4.i

for.inc5.loopexit.ur-lcssa.i:
  %17 = add i32 %.pr.i, 8
  %18 = add i32 %17, %7
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %for.inc5.i, label %for.body4.ur.i.preheader

for.body4.ur.i.preheader:
  %.ph = phi i32 [ %18, %for.inc5.loopexit.ur-lcssa.i ], [ %.pr.i, %for.body4.preheader.i ]
  br label %for.body4.ur.i

for.body4.ur.i:
  %20 = phi i32 [ %inc.ur.i, %for.body4.ur.i ], [ %.ph, %for.body4.ur.i.preheader ]
  %arrayidx.sum = add i32 %add.i, %20
  %arrayidx.ur.i = getelementptr inbounds i8, i8* %3, i32 %arrayidx.sum
  %21 = load i8, i8* %arrayidx.ur.i, align 1
  %conv.ur.i = sext i8 %21 to i32
  store i32 %conv.ur.i, i32* @c, align 4
  %inc.ur.i = add nsw i32 %20, 1
  store i32 %inc.ur.i, i32* @b, align 4
  %tobool3.ur.i = icmp eq i32 %inc.ur.i, 0
  br i1 %tobool3.ur.i, label %for.inc5.i.loopexit, label %for.body4.ur.i

for.inc5.i.loopexit:
  br label %for.inc5.i

for.inc5.i:
  %inc6.i = add nsw i32 %storemerge.i, 1
  br label %for.cond2thread-pre-split.i
}
