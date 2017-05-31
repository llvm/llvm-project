; RUN: opt < %s -partial-inliner -skip-partial-inlining-cost-analysis -S | FileCheck  %s
; RUN: opt < %s -passes=partial-inliner -skip-partial-inlining-cost-analysis -S | FileCheck   %s

%"class.base" = type { %"struct.base"* }
%"struct.base" = type opaque

@g = external local_unnamed_addr global i32, align 4

define i32 @callee_unknown_use2(i32 %arg) local_unnamed_addr #0 {
; CHECK-LABEL:define{{.*}}@callee_unknown_use2.{{[0-9]}}
; CHECK-NOT: alloca
; CHECK: call void @llvm.lifetime
bb:
  %tmp = alloca i32, align 4
  %tmp1 = bitcast i32* %tmp to i8*
  %tmp2 = load i32, i32* @g, align 4, !tbaa !2
  %tmp3 = add nsw i32 %tmp2, 1
  %tmp4 = icmp slt i32 %arg, 0
  br i1 %tmp4, label %bb6, label %bb5

bb5:                                              ; preds = %bb
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %tmp1) #2
  store i32 %tmp3, i32* %tmp, align 4, !tbaa !2
  store i32 %tmp3, i32* @g, align 4, !tbaa !2
  call void @bar(i32* nonnull %tmp) #2
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %tmp1) #2
  br label %bb6

bb6:                                              ; preds = %bb5, %bb
  %tmp7 = phi i32 [ 1, %bb5 ], [ 0, %bb ]
  %tmp10 = bitcast i8* %tmp1 to i32*
  ret i32 %tmp7
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare void @bar(i32*) local_unnamed_addr #2
declare void @bar2(i32*, i32*) local_unnamed_addr #1


; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define i32 @caller(i32 %arg) local_unnamed_addr #0 {
bb:
  %tmp = tail call i32 @callee_unknown_use2(i32 %arg)
  ret i32 %tmp
}

attributes #0 = { nounwind uwtable}
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 303574)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}



