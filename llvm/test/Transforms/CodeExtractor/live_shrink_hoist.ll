; RUN: opt -S -partial-inliner -max-num-inline-blocks=2 -skip-partial-inlining-cost-analysis  < %s |   FileCheck %s
; RUN: opt -S -passes=partial-inliner -max-num-inline-blocks=2  -skip-partial-inlining-cost-analysis < %s   | FileCheck %s

%class.A = type { i32 }

@cond = local_unnamed_addr global i32 0, align 4

; Function Attrs: uwtable
define void @_Z3foov() local_unnamed_addr  {
bb:
  %tmp = alloca %class.A, align 4
  %tmp1 = bitcast %class.A* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %tmp1) 
  %tmp2 = load i32, i32* @cond, align 4, !tbaa !2
  %tmp3 = icmp eq i32 %tmp2, 0
  br i1 %tmp3, label %bb4, label %bb9

bb4:                                              ; preds = %bb
  call void @_ZN1A7memfuncEv(%class.A* nonnull %tmp)
  %tmp5 = getelementptr inbounds %class.A, %class.A* %tmp, i64 0, i32 0
  %tmp6 = load i32, i32* %tmp5, align 4, !tbaa !6
  %tmp7 = icmp sgt i32 %tmp6, 0
  br i1 %tmp7, label %bb9, label %bb8

bb8:                                              ; preds = %bb4
  call void @_ZN1A7memfuncEv(%class.A* nonnull %tmp)
  br label %bb9

bb9:                                              ; preds = %bb8, %bb4, %bb
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %tmp1) 
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) 

declare void @_ZN1A7memfuncEv(%class.A*) local_unnamed_addr 

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) 

; Function Attrs: uwtable
define void @_Z3goov() local_unnamed_addr  {
bb:
  tail call void @_Z3foov()
  ret void
}

; CHECK-LABEL: define internal void @_Z3foov.1_
; CHECK: bb9:
; CHECK: call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %tmp1)
; CHECK:  br label %.exitStub



!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 304489)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !3, i64 0}
!7 = !{!"_ZTS1A", !3, i64 0}
