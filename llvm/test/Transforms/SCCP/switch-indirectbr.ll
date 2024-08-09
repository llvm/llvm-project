; RUN: opt -S -passes=ipsccp < %s | FileCheck %s
; CHECK:    switch i32 %state.0, label [[DEFAULT_UNREACHABLE:%.*]] [
; CHECK:       default.unreachable:
; CHECK-NEXT:    unreachable

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [23 x i8] c"OP_1:(instruction=%d)\0A\00", align 1
@.str.1 = private unnamed_addr constant [28 x i8] c"TERMINATE:(instruction=%d)\0A\00", align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() #0 {
entry:
  %bytecode = alloca [2 x ptr], align 16
  store ptr blockaddress(@main, %VM__OP_1), ptr %bytecode, align 16, !tbaa !5
  %arrayidx1 = getelementptr inbounds [2 x ptr], ptr %bytecode, i64 0, i64 1
  store ptr blockaddress(@main, %VM__TERMINATE), ptr %arrayidx1, align 8, !tbaa !5
  br label %while.body

while.body:                                       ; preds = %entry, %sw.epilog
  %state.0 = phi i32 [ 0, %entry ], [ %state.1, %sw.epilog ]
  %index.0 = phi i32 [ 0, %entry ], [ %index.2, %sw.epilog ]
  switch i32 %state.0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %VM__OP_1
    i32 2, label %sw.bb4
  ]

sw.bb:                                            ; preds = %while.body
  %idxprom = sext i32 %index.0 to i64
  %arrayidx2 = getelementptr inbounds [2 x ptr], ptr %bytecode, i64 0, i64 %idxprom
  %0 = load ptr, ptr %arrayidx2, align 8, !tbaa !5
  br label %indirectgoto

VM__OP_1:                                         ; preds = %while.body, %indirectgoto
  %index.1 = phi i32 [ %index.3, %indirectgoto ], [ %index.0, %while.body ]
  br label %sw.epilog

sw.bb4:                                           ; preds = %while.body
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %index.0)
  %inc = add nsw i32 %index.0, 1
  %idxprom5 = sext i32 %inc to i64
  %arrayidx6 = getelementptr inbounds [2 x ptr], ptr %bytecode, i64 0, i64 %idxprom5
  %1 = load ptr, ptr %arrayidx6, align 8, !tbaa !5
  br label %indirectgoto

sw.epilog:                                        ; preds = %while.body, %VM__OP_1
  %state.1 = phi i32 [ %state.0, %while.body ], [ 2, %VM__OP_1 ]
  %index.2 = phi i32 [ %index.0, %while.body ], [ %index.1, %VM__OP_1 ]
  br label %while.body, !llvm.loop !9

VM__TERMINATE:                                    ; preds = %indirectgoto
  %call7 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %index.3)
  ret i32 0

indirectgoto:                                     ; preds = %sw.bb4, %sw.bb
  %index.3 = phi i32 [ %inc, %sw.bb4 ], [ %index.0, %sw.bb ]
  %indirect.goto.dest = phi ptr [ %0, %sw.bb ], [ %1, %sw.bb4 ]
  indirectbr ptr %indirect.goto.dest, [label %VM__OP_1, label %VM__TERMINATE]
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr nocapture noundef readonly, ...) #1

attributes #0 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 18.0.0git (https://github.com/llvm/llvm-project.git 67782d2de5ea9c8653b8f0110237a3c355291c0e)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = distinct !{!9, !10, !11}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.unroll.disable"}
