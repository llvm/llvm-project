; RUN: opt -S -licm -max-dedicate-exit-iterations=5 < %s | FileCheck %s
; RUN: opt -S -licm -max-dedicate-exit-iterations=100 < %s | FileCheck %s -check-prefixes=SINK
; RUN: opt -S -passes='require<opt-remark-emit>,loop(licm)' -max-dedicate-exit-iterations=5 < %s | FileCheck %s
; RUN: opt -S -passes='require<opt-remark-emit>,loop(licm)' -max-dedicate-exit-iterations=100 < %s | FileCheck %s -check-prefixes=SINK
; Code sink in LICM requires the loop has dedicated exits. Use code sink
; in LICM to verify max-dedicate-exit-iterations is functioning.

@_ZL1m = internal global i64 0, align 8
@.str = private unnamed_addr constant [13 x i8] c"hello = %ld\0A\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_2.cc, i8* null }]

declare dso_local i64 @_Z3goov() local_unnamed_addr #0

; Function Attrs: argmemonly nounwind willreturn
declare {}* @llvm.invariant.start.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: uwtable
define dso_local void @_Z3fool(i64 %a) local_unnamed_addr {
entry:
  %cmp18.old.old.old.old.old = icmp slt i64 %a, 5
  %cmp14.old.old.old.old = icmp slt i64 %a, 4
  %cmp10.old.old.old = icmp slt i64 %a, 3
  %cmp6.old.old = icmp slt i64 %a, 2
  %cmp2.old = icmp slt i64 %a, 1
  %cmp = icmp slt i64 %a, 0
; Check load cannot be sinked out of loop when max-dedicate-exit-iterations=5
; and hasDedicatedExits return false, so it is hoisted out of loop.
; CHECK: load i64, i64* @_ZL1m, align 8
; CHECK: br label %do.body
; CHECK: do.body:
  br label %do.body

do.body:                                          ; preds = %do.body.backedge, %entry
  %t0 = load i64, i64* @_ZL1m, align 8, !tbaa !2
  %call = tail call i64 @_Z3goov()
  switch i64 %call, label %do.cond [
    i64 0, label %sw.bb
    i64 1, label %sw.bb1
    i64 2, label %sw.bb5
    i64 3, label %sw.bb9
    i64 4, label %sw.bb13
    i64 5, label %sw.bb17
  ]

sw.bb:                                            ; preds = %do.body
  br i1 %cmp, label %Done, label %do.body.backedge

sw.bb1:                                           ; preds = %do.body
  br i1 %cmp2.old, label %Done, label %do.body.backedge

sw.bb5:                                           ; preds = %do.body
  br i1 %cmp6.old.old, label %Done, label %do.body.backedge

sw.bb9:                                           ; preds = %do.body
  br i1 %cmp10.old.old.old, label %Done, label %do.body.backedge

sw.bb13:                                          ; preds = %do.body
  br i1 %cmp14.old.old.old.old, label %Done, label %do.body.backedge

sw.bb17:                                          ; preds = %do.body
  br i1 %cmp18.old.old.old.old.old, label %Done, label %do.body.backedge

do.cond:                                          ; preds = %do.body
  %cmp21.old.old.old.old.old.old = icmp slt i64 %call, 10000
  br i1 %cmp21.old.old.old.old.old.old, label %do.body.backedge, label %Done

do.body.backedge:                                 ; preds = %do.cond, %sw.bb17, %sw.bb13, %sw.bb9, %sw.bb5, %sw.bb1, %sw.bb
  br label %do.body

Done:                                             ; preds = %sw.bb, %sw.bb1, %sw.bb5, %sw.bb9, %sw.bb13, %sw.bb17, %do.cond
; Check load is sinked out of loop when max-dedicate-exit-iterations=100
; and hasDedicatedExits return true.
; SINK: Done:
; SINK: load i64, i64* @_ZL1m, align 8
; SINK-NEXT: tail call {{.*}} @printf
  %call22 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0), i64 %t0)
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_2.cc() section ".text.startup" {
entry:
  %call.i = tail call i64 @_Z3goov()
  store i64 %call.i, i64* @_ZL1m, align 8, !tbaa !2
  %t0 = tail call {}* @llvm.invariant.start.p0i8(i64 8, i8* bitcast (i64* @_ZL1m to i8*))
  ret void
}

attributes #0 = { readonly }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (trunk 372630) (llvm/trunk 372631)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
