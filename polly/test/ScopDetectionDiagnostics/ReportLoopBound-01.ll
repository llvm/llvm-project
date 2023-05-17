; RUN: opt %loadPolly \
; RUN:     -pass-remarks-missed="polly-detect" -polly-detect-track-failures \
; RUN:     -polly-allow-nonaffine-loops=false -polly-print-detect -disable-output \
; RUN:     < %s 2>&1| FileCheck %s --check-prefix=REJECTNONAFFINELOOPS
; RUN: opt %loadPolly \
; RUN:     -pass-remarks-missed="polly-detect" -polly-detect-track-failures \
; RUN:     -polly-allow-nonaffine-loops=true -polly-print-detect -disable-output \
; RUN:     < %s 2>&1| FileCheck %s --check-prefix=ALLOWNONAFFINELOOPS
; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" \
; RUN:     -polly-process-unprofitable=false \
; RUN:     -polly-detect-track-failures -polly-allow-nonaffine-loops=true \
; RUN:     -polly-allow-nonaffine -polly-print-detect -disable-output < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ALLOWNONAFFINEALL

; void f(int A[], int n) {
;   for (int i = 0; i < A[n+i]; i++)
;     A[i] = 0;
; }

; If we reject non-affine loops the non-affine loop bound will be reported:
;
; REJECTNONAFFINELOOPS: remark: ReportLoopBound-01.c:1:12: The following errors keep this region from being a Scop.
; REJECTNONAFFINELOOPS: remark: ReportLoopBound-01.c:2:8: Failed to derive an affine function from the loop bounds.
; REJECTNONAFFINELOOPS: remark: ReportLoopBound-01.c:3:5: Invalid Scop candidate ends here.

; If we allow non-affine loops the non-affine access will be reported:
;
; ALLOWNONAFFINELOOPS: remark: ReportLoopBound-01.c:1:12: The following errors keep this region from being a Scop.
; ALLOWNONAFFINELOOPS: remark: ReportLoopBound-01.c:3:5: The array subscript of "A" is not affine
; ALLOWNONAFFINELOOPS: remark: ReportLoopBound-01.c:3:5: Invalid Scop candidate ends here.

; If we allow non-affine loops and non-affine accesses the region will be reported as not profitable:
;
; ALLOWNONAFFINEALL: remark: ReportLoopBound-01.c:1:12: The following errors keep this region from being a Scop.
; ALLOWNONAFFINEALL: remark: ReportLoopBound-01.c:1:12: No profitable polyhedral optimization found
; ALLOWNONAFFINEALL: remark: ReportLoopBound-01.c:3:5: Invalid Scop candidate ends here.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i32 %n) !dbg !4 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata ptr %A, i64 0, metadata !13, metadata !DIExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata ptr %A, i64 0, metadata !13, metadata !DIExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata i32 %n, i64 0, metadata !15, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !18, metadata !DIExpression()), !dbg !20
  %idxprom = sext i32 %n to i64, !dbg !21
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %idxprom, !dbg !21
  %0 = load i32, ptr %arrayidx, align 4, !dbg !21
  %cmp3 = icmp sgt i32 %0, 0, !dbg !21
  br i1 %cmp3, label %for.body.lr.ph, label %for.end, !dbg !21

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body, !dbg !22

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvar = phi i64 [ 0, %for.body.lr.ph ], [ %indvar.next, %for.body ]
  %arrayidx2 = getelementptr i32, ptr %A, i64 %indvar, !dbg !24
  %1 = add i64 %indvar, 1, !dbg !24
  %inc = trunc i64 %1 to i32, !dbg !21
  store i32 0, ptr %arrayidx2, align 4, !dbg !24
  tail call void @llvm.dbg.value(metadata !{null}, i64 0, metadata !18, metadata !DIExpression()), !dbg !20
  %arrayidx3 = getelementptr inbounds i32, ptr %arrayidx, i64 %indvar, !dbg !21
  %2 = load i32, ptr %arrayidx3, align 4, !dbg !21
  %cmp = icmp slt i32 %inc, %2, !dbg !21
  %indvar.next = add i64 %indvar, 1, !dbg !21
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge, !dbg !21

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end, !dbg !25

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "ReportLoopBound-01.c", directory: "test/ScopDetectionDiagnostic/")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "ReportLoopBound-01.c", directory: "test/ScopDetectionDiagnostic/")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !9}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.6.0 "}
!13 = !DILocalVariable(name: "A", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!14 = !DILocation(line: 1, column: 12, scope: !4)
!15 = !DILocalVariable(name: "n", line: 1, arg: 2, scope: !4, file: !5, type: !9)
!16 = !DILocation(line: 1, column: 21, scope: !4)
!17 = !{i32 0}
!18 = !DILocalVariable(name: "i", line: 2, scope: !19, file: !5, type: !9)
!19 = distinct !DILexicalBlock(line: 2, column: 3, file: !1, scope: !4)
!20 = !DILocation(line: 2, column: 12, scope: !19)
!21 = !DILocation(line: 2, column: 8, scope: !19)
!22 = !DILocation(line: 2, column: 8, scope: !23)
!23 = distinct !DILexicalBlock(line: 2, column: 8, file: !1, scope: !19)
!24 = !DILocation(line: 3, column: 5, scope: !19)
!25 = !DILocation(line: 2, column: 8, scope: !26)
!26 = distinct !DILexicalBlock(line: 2, column: 8, file: !1, scope: !19)
!27 = !DILocation(line: 4, column: 1, scope: !4)
