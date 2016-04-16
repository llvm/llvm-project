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

define void @a(double* %A, i32 %n) #0 !dbg !4 {
entry:
  %cmp1 = icmp sgt i32 %n, 0, !dbg !10
  br i1 %cmp1, label %for.body.lr.ph, label %for.end, !dbg !10

for.body.lr.ph:                                   ; preds = %entry
  %0 = zext i32 %n to i64
  br label %for.body, !dbg !10

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvar = phi i64 [ 0, %for.body.lr.ph ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr double, double* %A, i64 %indvar, !dbg !12
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

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: 2, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "ReportFuncCall.c", directory: "/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics")
!2 = !{}
!4 = distinct !DISubprogram(name: "a", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "ReportFuncCall.c", directory: "/home/simbuerg/Projekte/llvm/tools/polly/test/ScopDetectionDiagnostics")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5.0 "}
!10 = !DILocation(line: 4, column: 8, scope: !11)
!11 = distinct !DILexicalBlock(line: 4, column: 3, file: !1, scope: !4)
!12 = !DILocation(line: 5, column: 12, scope: !13)
!13 = distinct !DILexicalBlock(line: 4, column: 27, file: !1, scope: !11)
!14 = !{!15, !15, i64 0}
!15 = !{!"double", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C/C++ TBAA"}
!18 = !DILocation(line: 7, column: 1, scope: !4)
