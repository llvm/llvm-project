; RUN: opt -S -passes=partial-inliner -skip-partial-inlining-cost-analysis < %s   | FileCheck %s

%class.A = type { i32 }
@cond = local_unnamed_addr global i32 0, align 4

; Function Attrs: uwtable
define void @_Z3foov() local_unnamed_addr  {
bb:
  %tmp = alloca %class.A, align 4
  %tmp1 = alloca %class.A, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %tmp)
  call void @llvm.lifetime.start.p0(ptr nonnull %tmp1)
  %tmp4 = load i32, ptr @cond, align 4, !tbaa !2
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb6, label %bb7

bb6:                                              ; preds = %bb
  call void @_ZN1A7memfuncEv(ptr nonnull %tmp)
  br label %bb7

bb7:                                              ; preds = %bb6, %bb
  call void @llvm.lifetime.end.p0(ptr nonnull %tmp1)
  call void @llvm.lifetime.end.p0(ptr nonnull %tmp)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(ptr nocapture)

declare void @_ZN1A7memfuncEv(ptr) local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(ptr nocapture)

; Function Attrs: uwtable
define void @_Z3goov() local_unnamed_addr  {
bb:
  tail call void @_Z3foov()
  ret void
}

; CHECK-LABEL: define internal void @_Z3foov.1.
; CHECK: newFuncRoot:
; CHECK-NEXT:  alloca
; CHECK-NEXT:  alloca
; CHECK-NEXT:  call void @llvm.lifetime.start.p0
; CHECK-NEXT:  call void @llvm.lifetime.start.p0
; CHECK:  call void @llvm.lifetime.end.p0
; CHECK-NEXT:  call void @llvm.lifetime.end.p0
; CHECK-NEXT:  br label {{.*}}exitStub


!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 304489)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
