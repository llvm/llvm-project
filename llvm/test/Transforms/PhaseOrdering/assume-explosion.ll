; RUN: opt -O3 -S < %s | FileCheck %s

; Confirm that we do not create assumes, clone them, 
; and then cause a compile-time explosion trying to 
; simplify them all. Ie, this can become nearly an 
; infinite-loop if things go bad.
; https://llvm.org/PR49785

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

@e = global i16 0, align 2
@a = global i32 0, align 4
@c = global i32 0, align 4
@b = global i32 0, align 4
@d = global i32 0, align 4

; Not checking complete IR because it could be very 
; large with vectorization and unrolling (thousands 
; of lines of IR).

define void @f() #0 {
; CHECK-LABEL: @f(
;
entry:
  store i32 5, i32* @c, align 4, !tbaa !3
  br label %for.cond

for.cond:
  %0 = load i32, i32* @c, align 4, !tbaa !3
  %cmp = icmp sle i32 %0, 63
  br i1 %cmp, label %for.body, label %for.end34

for.body:
  store i16 9, i16* @e, align 2, !tbaa !7
  br label %for.cond1

for.cond1:
  %1 = load i16, i16* @e, align 2, !tbaa !7
  %conv = zext i16 %1 to i32
  %cmp2 = icmp sle i32 %conv, 60
  br i1 %cmp2, label %for.body4, label %for.end32

for.body4:
  %2 = load i16, i16* @e, align 2, !tbaa !7
  %conv5 = zext i16 %2 to i32
  %3 = load i32, i32* @b, align 4, !tbaa !3
  %xor = xor i32 %conv5, %3
  %4 = load i32, i32* @d, align 4, !tbaa !3
  %cmp6 = icmp ne i32 %xor, %4
  br i1 %cmp6, label %if.then, label %if.end27

if.then:
  %5 = load i32, i32* @a, align 4, !tbaa !3
  %conv8 = sext i32 %5 to i64
  %6 = inttoptr i64 %conv8 to i8*
  store i8 3, i8* %6, align 1, !tbaa !9
  br label %for.cond9

for.cond9:
  %7 = load i8, i8* %6, align 1, !tbaa !9
  %conv10 = sext i8 %7 to i32
  %cmp11 = icmp sle i32 %conv10, 32
  br i1 %cmp11, label %for.body13, label %for.end26

for.body13:
  %8 = load i8, i8* %6, align 1, !tbaa !9
  %tobool = icmp ne i8 %8, 0
  br i1 %tobool, label %if.then14, label %if.end

if.then14:
  store i8 1, i8* bitcast (i32* @a to i8*), align 1, !tbaa !9
  br label %for.cond15

for.cond15:
  %9 = load i8, i8* bitcast (i32* @a to i8*), align 1, !tbaa !9
  %conv16 = sext i8 %9 to i32
  %cmp17 = icmp sle i32 %conv16, 30
  br i1 %cmp17, label %for.body19, label %for.end

for.body19:
  %10 = load i32, i32* @c, align 4, !tbaa !3
  %cmp20 = icmp eq i32 0, %10
  %conv21 = zext i1 %cmp20 to i32
  %11 = load i8, i8* bitcast (i32* @a to i8*), align 1, !tbaa !9
  %conv22 = sext i8 %11 to i32
  %and = and i32 %conv22, %conv21
  %conv23 = trunc i32 %and to i8
  store i8 %conv23, i8* bitcast (i32* @a to i8*), align 1, !tbaa !9
  br label %for.cond15, !llvm.loop !10

for.end:
  br label %if.end

if.end:
  br label %for.inc

for.inc:
  %12 = load i8, i8* %6, align 1, !tbaa !9
  %conv24 = sext i8 %12 to i32
  %add = add nsw i32 %conv24, 1
  %conv25 = trunc i32 %add to i8
  store i8 %conv25, i8* %6, align 1, !tbaa !9
  br label %for.cond9, !llvm.loop !12

for.end26:
  br label %if.end27

if.end27:
  br label %for.inc28

for.inc28:
  %13 = load i16, i16* @e, align 2, !tbaa !7
  %conv29 = zext i16 %13 to i32
  %add30 = add nsw i32 %conv29, 1
  %conv31 = trunc i32 %add30 to i16
  store i16 %conv31, i16* @e, align 2, !tbaa !7
  br label %for.cond1, !llvm.loop !13

for.end32:
  br label %for.inc33

for.inc33:
  %14 = load i32, i32* @c, align 4, !tbaa !3
  %inc = add nsw i32 %14, 1
  store i32 %inc, i32* @c, align 4, !tbaa !3
  br label %for.cond, !llvm.loop !14

for.end34:
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind ssp uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 7a4abc07dd8f1d8217e482ebbf438197c1aea7f0)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !5, i64 0}
!9 = !{!5, !5, i64 0}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
!13 = distinct !{!13, !11}
!14 = distinct !{!14, !11}
