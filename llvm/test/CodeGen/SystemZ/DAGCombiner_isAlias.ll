; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s  | FileCheck %s
;
; Check that the second load of @g_2 is not incorrectly eliminated by
; DAGCombiner. It is needed since the preceding store is aliasing.

; %.b1.i = load i1, i1* @g_2, align 4
; ...
; %g_717.sink.i = select i1 %cmp.i, i1* @g_717, i1* @g_2
; store i1 true, i1* %g_717.sink.i, align 4
; %.b = load i1, i1* @g_2, align 4

; CHECK: # %bb.6: # %crc32_gentab.exit
; CHECK:        larl    %r2, g_2
; CHECK-NEXT:   llc     %r3, 0(%r2)
; CHECK-NOT:    %r2
; CHECK:        llc     %r1, 0(%r2)

@g_2 = external hidden unnamed_addr global i1, align 4
@.str.1 = external hidden unnamed_addr constant [4 x i8], align 2
@g_717 = external hidden unnamed_addr global i1, align 4
@.str.2 = external hidden unnamed_addr constant [6 x i8], align 2
@crc32_context = external hidden unnamed_addr global i32, align 4
@crc32_tab = external hidden unnamed_addr global [256 x i32], align 4
@g_5 = external hidden unnamed_addr global i32, align 4
@.str.4 = external hidden unnamed_addr constant [15 x i8], align 2

; Function Attrs: nounwind
define signext i32 @main(i32 signext %argc, i8** nocapture readonly %argv) local_unnamed_addr #0 {
entry:
  %cmp = icmp eq i32 %argc, 2
  br i1 %cmp, label %cond.true, label %vector.ph

cond.true:                                        ; preds = %entry
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !2
  %1 = load i8, i8* %0, align 1, !tbaa !6
  %conv4 = zext i8 %1 to i32
  %sub = sub nsw i32 49, %conv4
  %cmp8 = icmp eq i32 %sub, 0
  br i1 %cmp8, label %if.then, label %if.end35

if.then:                                          ; preds = %cond.true
  %arrayidx11 = getelementptr inbounds i8, i8* %0, i64 1
  %2 = load i8, i8* %arrayidx11, align 1, !tbaa !6
  %conv12 = zext i8 %2 to i32
  %sub13 = sub nsw i32 0, %conv12
  br label %if.end35

if.end35:                                         ; preds = %if.then, %cond.true
  %__result.0 = phi i32 [ %sub13, %if.then ], [ %sub, %cond.true ]
  %phitmp = icmp eq i32 %__result.0, 0
  %spec.select = zext i1 %phitmp to i32
  br label %vector.ph

vector.ph:                                        ; preds = %if.end35, %entry
  %print_hash_value.0 = phi i32 [ 0, %entry ], [ %spec.select, %if.end35 ]
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind22 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %vector.ph ], [ %vec.ind.next23, %vector.body ]
  %3 = and <4 x i32> %vec.ind22, <i32 1, i32 1, i32 1, i32 1>
  %4 = icmp eq <4 x i32> %3, zeroinitializer
  %5 = lshr <4 x i32> %vec.ind22, <i32 1, i32 1, i32 1, i32 1>
  %6 = xor <4 x i32> %5, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %7 = select <4 x i1> %4, <4 x i32> %5, <4 x i32> %6
  %8 = and <4 x i32> %7, <i32 1, i32 1, i32 1, i32 1>
  %9 = icmp eq <4 x i32> %8, zeroinitializer
  %10 = lshr <4 x i32> %7, <i32 1, i32 1, i32 1, i32 1>
  %11 = xor <4 x i32> %10, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %12 = select <4 x i1> %9, <4 x i32> %10, <4 x i32> %11
  %13 = and <4 x i32> %12, <i32 1, i32 1, i32 1, i32 1>
  %14 = icmp eq <4 x i32> %13, zeroinitializer
  %15 = lshr <4 x i32> %12, <i32 1, i32 1, i32 1, i32 1>
  %16 = xor <4 x i32> %15, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %17 = select <4 x i1> %14, <4 x i32> %15, <4 x i32> %16
  %18 = and <4 x i32> %17, <i32 1, i32 1, i32 1, i32 1>
  %19 = icmp eq <4 x i32> %18, zeroinitializer
  %20 = lshr <4 x i32> %17, <i32 1, i32 1, i32 1, i32 1>
  %21 = xor <4 x i32> %20, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %22 = select <4 x i1> %19, <4 x i32> %20, <4 x i32> %21
  %23 = and <4 x i32> %22, <i32 1, i32 1, i32 1, i32 1>
  %24 = icmp eq <4 x i32> %23, zeroinitializer
  %25 = lshr <4 x i32> %22, <i32 1, i32 1, i32 1, i32 1>
  %26 = xor <4 x i32> %25, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %27 = select <4 x i1> %24, <4 x i32> %25, <4 x i32> %26
  %28 = and <4 x i32> %27, <i32 1, i32 1, i32 1, i32 1>
  %29 = icmp eq <4 x i32> %28, zeroinitializer
  %30 = lshr <4 x i32> %27, <i32 1, i32 1, i32 1, i32 1>
  %31 = xor <4 x i32> %30, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %32 = select <4 x i1> %29, <4 x i32> %30, <4 x i32> %31
  %33 = and <4 x i32> %32, <i32 1, i32 1, i32 1, i32 1>
  %34 = icmp eq <4 x i32> %33, zeroinitializer
  %35 = lshr <4 x i32> %32, <i32 1, i32 1, i32 1, i32 1>
  %36 = xor <4 x i32> %35, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %37 = select <4 x i1> %34, <4 x i32> %35, <4 x i32> %36
  %38 = and <4 x i32> %37, <i32 1, i32 1, i32 1, i32 1>
  %39 = icmp eq <4 x i32> %38, zeroinitializer
  %40 = lshr <4 x i32> %37, <i32 1, i32 1, i32 1, i32 1>
  %41 = xor <4 x i32> %40, <i32 -306674912, i32 -306674912, i32 -306674912, i32 -306674912>
  %42 = select <4 x i1> %39, <4 x i32> %40, <4 x i32> %41
  %43 = getelementptr inbounds [256 x i32], [256 x i32]* @crc32_tab, i64 0, i64 %index
  %44 = bitcast i32* %43 to <4 x i32>*
  store <4 x i32> %42, <4 x i32>* %44, align 4, !tbaa !7
  %index.next = add i64 %index, 4
  %vec.ind.next23 = add <4 x i32> %vec.ind22, <i32 4, i32 4, i32 4, i32 4>
  %45 = icmp eq i64 %index.next, 256
  br i1 %45, label %crc32_gentab.exit, label %vector.body

crc32_gentab.exit:                                ; preds = %vector.body
  %46 = load i32, i32* @g_5, align 4, !tbaa !7
  %.b1.i = load i1, i1* @g_2, align 4
  %47 = select i1 %.b1.i, i32 1, i32 2
  %and.i21 = and i32 %47, %46
  store i32 %and.i21, i32* @g_5, align 4, !tbaa !7
  %cmp.i = icmp eq i32 %and.i21, 1
  %g_717.sink.i = select i1 %cmp.i, i1* @g_717, i1* @g_2
  store i1 true, i1* %g_717.sink.i, align 4
  %.b = load i1, i1* @g_2, align 4
  %conv44 = select i1 %.b, i64 1, i64 2
  tail call fastcc void @transparent_crc(i64 %conv44, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), i32 signext %print_hash_value.0)
  %.b20 = load i1, i1* @g_717, align 4
  %conv45 = select i1 %.b20, i64 2, i64 0
  tail call fastcc void @transparent_crc(i64 %conv45, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.2, i64 0, i64 0), i32 signext %print_hash_value.0)
  %48 = load i32, i32* @crc32_context, align 4, !tbaa !7
  %49 = xor i32 %48, -1
  %call.i = tail call signext i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.4, i64 0, i64 0), i32 zeroext %49) #2
  ret i32 0
}

; Function Attrs: nounwind
declare hidden fastcc void @transparent_crc(i64, i8*, i32 signext) unnamed_addr #0

; Function Attrs: nounwind
declare signext i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #1

!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!4, !4, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !4, i64 0}
