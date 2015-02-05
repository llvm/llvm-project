; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1 | FileCheck %s

; #define N 1024
; double invalidCall(double A[N]);
;
; void a(double A[N], int n) {
;   for (int i=0; i<n; ++i) {
;     A[i] = invalidCall(A);
;   }
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @a(double* %A, i32 %n) #0 {
entry:
  %cmp1 = icmp sgt i32 %n, 0, !dbg !10
  br i1 %cmp1, label %for.body.lr.ph, label %for.end, !dbg !10

for.body.lr.ph:                                   ; preds = %entry
  %0 = zext i32 %n to i64
  br label %for.body, !dbg !10

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvar = phi i64 [ 0, %for.body.lr.ph ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr double* %A, i64 %indvar, !dbg !12
  %call = tail call double @invalidCall(double* %A) #2, !dbg !12
  store double %call, double* %arrayidx, align 8, !dbg !12, !tbaa !14
  %indvar.next = add i64 %indvar, 1, !dbg !10
  %exitcond = icmp eq i64 %indvar.next, %0, !dbg !10
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !10

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !18
}

declare double @invalidCall(double*) #1

; CHECK: remark: ReportFuncCall.c:4:8: The following errors keep this region from being a Scop.
; CHECK: remark: ReportFuncCall.c:5:12: This function call cannot be handled. Try to inline it.

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{!"0x11\0012\00clang version 3.5.0 \001\00\000\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics/ReportFuncCall.c] [DW_LANG_C99]
!1 = !{!"ReportFuncCall.c", !"/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00a\00a\00\003\000\001\000\006\00256\001\003", !1, !5, !6, null, void (double*, i32)* @a, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [a]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics/ReportFuncCall.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.5.0 "}
!10 = !MDLocation(line: 4, column: 8, scope: !11)
!11 = !{!"0xb\004\003\000", !1, !4} ; [ DW_TAG_lexical_block ] [/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics/ReportFuncCall.c]
!12 = !MDLocation(line: 5, column: 12, scope: !13)
!13 = !{!"0xb\004\0027\000", !1, !11} ; [ DW_TAG_lexical_block ] [/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics/ReportFuncCall.c]
!14 = !{!15, !15, i64 0}
!15 = !{!"double", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C/C++ TBAA"}
!18 = !MDLocation(line: 7, column: 1, scope: !4)
