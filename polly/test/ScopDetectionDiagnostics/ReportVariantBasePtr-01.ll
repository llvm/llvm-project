; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s

; struct b {
;   double **b;
; };
;
; void a(struct b *A) {
;   for (int i=0; i<32; i++)
;     A[i].b[i] = 0;
; }

; The loads are currently just adds %7 to the list of required invariant loads
; and only -polly-scops checks whether it is actionally possible the be load
; hoisted. The SCoP is still rejected by -polly-detect because it may alias
; with %A and is not considered to be eligble for runtime alias checking.

; CHECK: remark: ReportVariantBasePtr01.c:6:8: The following errors keep this region from being a Scop.
; CHECK: remark: ReportVariantBasePtr01.c:7:5: Accesses to the arrays "A", " <unknown> " may access the same memory.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.b = type { double** }

define void @a(%struct.b* nocapture readonly %A) #0 !dbg !4 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  tail call void @llvm.dbg.value(metadata %struct.b* %A, i64 0, metadata !16, metadata !DIExpression()), !dbg !23
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !17, metadata !DIExpression()), !dbg !25
  br label %for.body, !dbg !27

for.body:                                         ; preds = %for.body, %entry.split
  %indvar4 = phi i64 [ %indvar.next, %for.body ], [ 0, %entry.split ]
  %b = getelementptr inbounds %struct.b, %struct.b* %A, i64 %indvar4, i32 0, !dbg !26
  %0 = mul i64 %indvar4, 4, !dbg !26
  %1 = add i64 %0, 3, !dbg !26
  %2 = add i64 %0, 2, !dbg !26
  %3 = add i64 %0, 1, !dbg !26
  %4 = load double**, double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx = getelementptr double*, double** %4, i64 %0, !dbg !26
  store double* null, double** %arrayidx, align 8, !dbg !26, !tbaa !33
  %5 = load double**, double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx.1 = getelementptr double*, double** %5, i64 %3, !dbg !26
  store double* null, double** %arrayidx.1, align 8, !dbg !26, !tbaa !33
  %6 = load double**, double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx.2 = getelementptr double*, double** %6, i64 %2, !dbg !26
  store double* null, double** %arrayidx.2, align 8, !dbg !26, !tbaa !33
  %7 = load double**, double*** %b, align 8, !dbg !26, !tbaa !28
  %arrayidx.3 = getelementptr double*, double** %7, i64 %1, !dbg !26
  store double* null, double** %arrayidx.3, align 8, !dbg !26, !tbaa !33
  %indvar.next = add i64 %indvar4, 1, !dbg !27
  %exitcond = icmp eq i64 %indvar.next, 8, !dbg !27
  br i1 %exitcond, label %for.end, label %for.body, !dbg !27

for.end:                                          ; preds = %for.body
  ret void, !dbg !34
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "ReportVariantBasePtr01.c", directory: "test/ScopDetectionDiagnostics")
!2 = !{}
!4 = distinct !DISubprogram(name: "a", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 5, file: !1, scope: !5, type: !6, variables: !15)
!5 = !DIFile(filename: "ReportVariantBasePtr01.c", directory: "test/ScopDetectionDiagnostics")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "b", line: 1, size: 64, align: 64, file: !1, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 2, size: 64, align: 64, file: !1, scope: !9, baseType: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !13)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !14)
!14 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!15 = !{!16, !17}
!16 = !DILocalVariable(name: "A", line: 5, arg: 1, scope: !4, file: !5, type: !8)
!17 = !DILocalVariable(name: "i", line: 6, scope: !18, file: !5, type: !19)
!18 = distinct !DILexicalBlock(line: 6, column: 3, file: !1, scope: !4)
!19 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{!"clang version 3.5.0 "}
!23 = !DILocation(line: 5, column: 18, scope: !4)
!24 = !{i32 0}
!25 = !DILocation(line: 6, column: 12, scope: !18)
!26 = !DILocation(line: 7, column: 5, scope: !18)
!27 = !DILocation(line: 6, column: 8, scope: !18)
!28 = !{!29, !30, i64 0}
!29 = !{!"b", !30, i64 0}
!30 = !{!"any pointer", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !{!30, !30, i64 0}
!34 = !DILocation(line: 8, column: 1, scope: !4)
