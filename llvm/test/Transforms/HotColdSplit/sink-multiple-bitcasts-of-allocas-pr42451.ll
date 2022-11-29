; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=-1 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

@c = common global i32 0, align 4
@h = common global i32 0, align 4

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1

declare i32* @m()

; CHECK-LABEL: define void @main()
; CHECK-NEXT: bb:
; CHECK-NEXT:   %.sroa.4.i = alloca [20 x i8], align 2
; CHECK-NEXT:   %.sroa.5.i = alloca [6 x i8], align 8
; CHECK-NEXT:   %i = bitcast [6 x i8]* %.sroa.5.i to i8*

define void @main() #2 {
bb:
  %.sroa.4.i = alloca [20 x i8], align 2
  %.sroa.5.i = alloca [6 x i8], align 8
  %i = bitcast [6 x i8]* %.sroa.5.i to i8*
  %i1 = load i32, i32* @h, align 4, !tbaa !4
  %i2 = icmp ne i32 %i1, 0
  br i1 %i2, label %bb11, label %bb3

bb3:                                              ; preds = %bb
  %i4 = call i32* @m()
  %.sroa.4.0..sroa_idx21.i = getelementptr inbounds [20 x i8], [20 x i8]* %.sroa.4.i, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 20, i8* %.sroa.4.0..sroa_idx21.i)
  %.sroa.5.0..sroa_idx16.i = getelementptr inbounds [6 x i8], [6 x i8]* %.sroa.5.i, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 6, i8* %.sroa.5.0..sroa_idx16.i)
  call void @llvm.memset.p0i8.i64(i8* align 2 %.sroa.4.0..sroa_idx21.i, i8 0, i64 20, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 8 %.sroa.5.0..sroa_idx16.i, i8 0, i64 6, i1 false)
  %i5 = load i32, i32* @c, align 4, !tbaa !4
  %i6 = trunc i32 %i5 to i16
  call void @llvm.lifetime.end.p0i8(i64 20, i8* %.sroa.4.0..sroa_idx21.i)
  call void @llvm.lifetime.end.p0i8(i64 6, i8* %.sroa.5.0..sroa_idx16.i)
  call void @llvm.lifetime.start.p0i8(i64 6, i8* %i)
  call void @llvm.memset.p0i8.i64(i8* align 1 %i, i8 3, i64 6, i1 false)
  br label %bb7

bb7:                                              ; preds = %bb7, %bb3
  %.0.i = phi i32 [ 0, %bb3 ], [ %i9, %bb7 ]
  %i8 = sext i32 %.0.i to i64
  %i9 = add nsw i32 %.0.i, 1
  %i10 = icmp slt i32 %i9, 6
  br i1 %i10, label %bb7, label %l.exit

l.exit:                                           ; preds = %bb7
  call void @llvm.lifetime.end.p0i8(i64 6, i8* %i)
  br label %bb11

bb11:                                             ; preds = %l.exit, %bb
  %i12 = phi i1 [ true, %bb ], [ true, %l.exit ]
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { cold }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 14]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{!"Apple clang version 11.0.0 (clang-1100.0.20.17)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
