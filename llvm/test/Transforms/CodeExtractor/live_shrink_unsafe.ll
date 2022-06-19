; The expected behavior of this file is expected to change when partial
; inlining legality check is enhanced.

; RUN: opt -S -partial-inliner -skip-partial-inlining-cost-analysis  < %s   | FileCheck %s
; RUN: opt -S -passes=partial-inliner -skip-partial-inlining-cost-analysis < %s |   FileCheck %s

%class.A = type { i32 }

@cond = local_unnamed_addr global i32 0, align 4
@condptr = external local_unnamed_addr global i32*, align 8

; Function Attrs: uwtable
define void @_Z3foo_unknown_mem_accessv() local_unnamed_addr  {
bb:
  %tmp = alloca %class.A, align 4
  %tmp1 = alloca %class.A, align 4
  %tmp2 = bitcast %class.A* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %tmp2) 
  %tmp3 = bitcast %class.A* %tmp1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %tmp3) 
  %tmp4 = load i32*, i32** @condptr, align 8, !tbaa !2
  %tmp5 = load i32, i32* %tmp4, align 4, !tbaa !6
  %tmp6 = icmp eq i32 %tmp5, 0
  br i1 %tmp6, label %bb7, label %bb8

bb7:                                              ; preds = %bb
  call void @_ZN1A7memfuncEv(%class.A* nonnull %tmp)
  br label %bb8

bb8:                                              ; preds = %bb7, %bb
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %tmp3) 
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %tmp2) 
  ret void
}

declare void @_Z3barv() local_unnamed_addr
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) 
declare void @_ZN1A7memfuncEv(%class.A*) local_unnamed_addr 
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) 

define void @_Z3foo_unknown_calli(i32 %arg) local_unnamed_addr {
bb:
  %tmp = alloca %class.A, align 4
  %tmp1 = bitcast %class.A* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %tmp1) 
  tail call void @_Z3barv()
  %tmp2 = icmp eq i32 %arg, 0
  br i1 %tmp2, label %bb3, label %bb4

bb3:                                              ; preds = %bb
  call void @_ZN1A7memfuncEv(%class.A* nonnull %tmp)
  br label %bb4

bb4:                                              ; preds = %bb3, %bb
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %tmp1) 
  ret void
}

define void @_Z3goov() local_unnamed_addr  {
; CHECK-LABEL: @_Z3goov
; CHECK-NEXT: bb:
; CHECK: alloca
; CHECK: lifetime
bb:
  call void @_Z3foo_unknown_mem_accessv()
  %tmp = load i32, i32* @cond, align 4, !tbaa !2
  tail call void @_Z3foo_unknown_calli(i32 %tmp)
  ret void
}

; CHECK-LABEL: define internal void @_Z3foo_unknown_calli.1.bb3
; CHECK: newFuncRoot:
; CHECK-NEXT: br label %bb3

; CHECK: bb3:
; CHECK-NOT: lifetime.ed
; CHECK: br label %bb4.exitStub

; CHECK: bb4.exitStub:
; CHECK-NEXT: ret void



!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 304489)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}
