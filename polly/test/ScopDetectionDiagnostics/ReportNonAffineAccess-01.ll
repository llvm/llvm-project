; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

; void f(int A[]) {
;   for(int i=0; i<42; ++i)
;     A[i*i] = 0;
; }


; CHECK: remark: ReportNonAffineAccess-01.c:2:7: The following errors keep this region from being a Scop.
; CHECK: remark: ReportNonAffineAccess-01.c:3:5: The array subscript of "A" is not affine
; CHECK: remark: ReportNonAffineAccess-01.c:3:5: Invalid Scop candidate ends here.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) !dbg !4 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata i32* %A, i64 0, metadata !13, metadata !DIExpression()), !dbg !14
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !16, metadata !DIExpression()), !dbg !18
  br label %for.body, !dbg !19

for.body:                                         ; preds = %entry.split, %for.body
  %0 = phi i32 [ 0, %entry.split ], [ %1, %for.body ], !dbg !20
  %mul = mul nsw i32 %0, %0, !dbg !20
  %idxprom1 = zext i32 %mul to i64, !dbg !20
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom1, !dbg !20
  store i32 0, i32* %arrayidx, align 4, !dbg !20
  %1 = add nsw i32 %0, 1, !dbg !21
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !16, metadata !DIExpression()), !dbg !18
  %exitcond = icmp ne i32 %1, 42, !dbg !19
  br i1 %exitcond, label %for.body, label %for.end, !dbg !19

for.end:                                          ; preds = %for.body
  ret void, !dbg !22
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "ReportNonAffineAccess-01.c", directory: "test/ScopDetectionDiagnostic/")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "ReportNonAffineAccess-01.c", directory: "test/ScopDetectionDiagnostic/")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.6.0 "}
!13 = !DILocalVariable(name: "A", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!14 = !DILocation(line: 1, column: 12, scope: !4)
!15 = !{i32 0}
!16 = !DILocalVariable(name: "i", line: 2, scope: !17, file: !5, type: !9)
!17 = distinct !DILexicalBlock(line: 2, column: 3, file: !1, scope: !4)
!18 = !DILocation(line: 2, column: 11, scope: !17)
!19 = !DILocation(line: 2, column: 7, scope: !17)
!20 = !DILocation(line: 3, column: 5, scope: !17)
!21 = !DILocation(line: 2, column: 22, scope: !17)
!22 = !DILocation(line: 4, column: 1, scope: !4)
