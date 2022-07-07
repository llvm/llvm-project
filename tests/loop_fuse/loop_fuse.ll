; ModuleID = 'loop_fuse.c'
source_filename = "loop_fuse.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local void @init(i32* noundef %a, i32* noundef %b, i32* noundef %c, i32 noundef %n) #0 {
entry:
  %a.addr = alloca i32*, align 8
  %b.addr = alloca i32*, align 8
  %c.addr = alloca i32*, align 8
  %n.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %i3 = alloca i32, align 4
  %i17 = alloca i32, align 4
  %i31 = alloca i32, align 4
  store i32* %a, i32** %a.addr, align 8
  store i32* %b, i32** %b.addr, align 8
  store i32* %c, i32** %c.addr, align 8
  store i32 %n, i32* %n.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %n.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4
  %3 = load i32, i32* %i, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32*, i32** %c.addr, align 8
  %5 = load i32, i32* %i, align 4
  %idxprom = sext i32 %5 to i64
  %arrayidx = getelementptr inbounds i32, i32* %4, i64 %idxprom
  store i32 %add, i32* %arrayidx, align 4
  %6 = load i32, i32* %i, align 4
  %7 = load i32, i32* %i, align 4
  %mul = mul nsw i32 %6, %7
  %8 = load i32*, i32** %b.addr, align 8
  %9 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %9 to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %8, i64 %idxprom1
  store i32 %mul, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %10 = load i32, i32* %i, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond, !llvm.loop !4

for.end:                                          ; preds = %for.cond
  store i32 0, i32* %i3, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc14, %for.end
  %11 = load i32, i32* %i3, align 4
  %cmp5 = icmp slt i32 %11, 10
  br i1 %cmp5, label %for.body6, label %for.end16

for.body6:                                        ; preds = %for.cond4
  %12 = load i32*, i32** %b.addr, align 8
  %13 = load i32, i32* %i3, align 4
  %idxprom7 = sext i32 %13 to i64
  %arrayidx8 = getelementptr inbounds i32, i32* %12, i64 %idxprom7
  %14 = load i32, i32* %arrayidx8, align 4
  %15 = load i32*, i32** %c.addr, align 8
  %16 = load i32, i32* %i3, align 4
  %idxprom9 = sext i32 %16 to i64
  %arrayidx10 = getelementptr inbounds i32, i32* %15, i64 %idxprom9
  %17 = load i32, i32* %arrayidx10, align 4
  %add11 = add nsw i32 %14, %17
  %18 = load i32*, i32** %a.addr, align 8
  %19 = load i32, i32* %i3, align 4
  %idxprom12 = sext i32 %19 to i64
  %arrayidx13 = getelementptr inbounds i32, i32* %18, i64 %idxprom12
  store i32 %add11, i32* %arrayidx13, align 4
  br label %for.inc14

for.inc14:                                        ; preds = %for.body6
  %20 = load i32, i32* %i3, align 4
  %inc15 = add nsw i32 %20, 1
  store i32 %inc15, i32* %i3, align 4
  br label %for.cond4, !llvm.loop !6

for.end16:                                        ; preds = %for.cond4
  store i32 0, i32* %i17, align 4
  br label %for.cond18

for.cond18:                                       ; preds = %for.inc28, %for.end16
  %21 = load i32, i32* %i17, align 4
  %cmp19 = icmp slt i32 %21, 10
  br i1 %cmp19, label %for.body20, label %for.end30

for.body20:                                       ; preds = %for.cond18
  %22 = load i32*, i32** %b.addr, align 8
  %23 = load i32, i32* %i17, align 4
  %idxprom21 = sext i32 %23 to i64
  %arrayidx22 = getelementptr inbounds i32, i32* %22, i64 %idxprom21
  %24 = load i32, i32* %arrayidx22, align 4
  %25 = load i32*, i32** %c.addr, align 8
  %26 = load i32, i32* %i17, align 4
  %idxprom23 = sext i32 %26 to i64
  %arrayidx24 = getelementptr inbounds i32, i32* %25, i64 %idxprom23
  %27 = load i32, i32* %arrayidx24, align 4
  %add25 = add nsw i32 %24, %27
  %28 = load i32*, i32** %a.addr, align 8
  %29 = load i32, i32* %i17, align 4
  %idxprom26 = sext i32 %29 to i64
  %arrayidx27 = getelementptr inbounds i32, i32* %28, i64 %idxprom26
  store i32 %add25, i32* %arrayidx27, align 4
  br label %for.inc28

for.inc28:                                        ; preds = %for.body20
  %30 = load i32, i32* %i17, align 4
  %inc29 = add nsw i32 %30, 1
  store i32 %inc29, i32* %i17, align 4
  br label %for.cond18, !llvm.loop !7

for.end30:                                        ; preds = %for.cond18
  store i32 0, i32* %i31, align 4
  br label %for.cond32

for.cond32:                                       ; preds = %for.inc42, %for.end30
  %31 = load i32, i32* %i31, align 4
  %32 = load i32, i32* %n.addr, align 4
  %cmp33 = icmp slt i32 %31, %32
  br i1 %cmp33, label %for.body34, label %for.end44

for.body34:                                       ; preds = %for.cond32
  %33 = load i32*, i32** %b.addr, align 8
  %34 = load i32, i32* %i31, align 4
  %idxprom35 = sext i32 %34 to i64
  %arrayidx36 = getelementptr inbounds i32, i32* %33, i64 %idxprom35
  %35 = load i32, i32* %arrayidx36, align 4
  %36 = load i32*, i32** %c.addr, align 8
  %37 = load i32, i32* %i31, align 4
  %idxprom37 = sext i32 %37 to i64
  %arrayidx38 = getelementptr inbounds i32, i32* %36, i64 %idxprom37
  %38 = load i32, i32* %arrayidx38, align 4
  %add39 = add nsw i32 %35, %38
  %39 = load i32*, i32** %a.addr, align 8
  %40 = load i32, i32* %i31, align 4
  %idxprom40 = sext i32 %40 to i64
  %arrayidx41 = getelementptr inbounds i32, i32* %39, i64 %idxprom40
  store i32 %add39, i32* %arrayidx41, align 4
  br label %for.inc42

for.inc42:                                        ; preds = %for.body34
  %41 = load i32, i32* %i31, align 4
  %inc43 = add nsw i32 %41, 1
  store i32 %inc43, i32* %i31, align 4
  br label %for.cond32, !llvm.loop !8

for.end44:                                        ; preds = %for.cond32
  ret void
}

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 14.0.6 (https://github.com/shravankumar0811/llvm-project.git 47ee914ea16086c1958b93540ed2351bcdae7cdb)"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
!6 = distinct !{!6, !5}
!7 = distinct !{!7, !5}
!8 = distinct !{!8, !5}
