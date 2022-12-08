; RUN: opt -S -passes=partial-inliner  -skip-partial-inlining-cost-analysis  < %s |   FileCheck %s
; RUN: opt -S -passes=partial-inliner  -skip-partial-inlining-cost-analysis  < %s   | FileCheck %s

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
  br i1 %tmp3, label %bb4, label %bb5

bb4:                                              ; preds = %bb
  call void @_ZN1A7memfuncEv(%class.A* nonnull %tmp)
  br label %bb5

bb5:                                              ; preds = %bb4, %bb
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
; CHECK-LABEL: @_Z3goov()
bb:
; CHECK: bb:
; CHECK-NOT: alloca
; CHECK-NOT: bitcast
; CHECK-NOT: llvm.lifetime
; CHECK: br i1
; CHECK: codeRepl.i:
; CHECK: call void @_Z3foov.1.

  tail call void @_Z3foov()
  ret void
}

; CHECK-LABEL: define internal void @_Z3foov.1.
; CHECK: newFuncRoot:
; CHECK-NEXT:  %tmp = alloca %class.A
; CHECK-NEXT:  %tmp1 = bitcast %class.A* %tmp to i8*
; CHECK-NEXT:  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %tmp1)
; CHECK:  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %tmp1)
; CHECK-NEXT:  br label %bb5.exitStub


!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 304489)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
