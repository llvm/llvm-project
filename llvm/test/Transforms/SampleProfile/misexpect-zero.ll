; Ensure MisExpect does not crash when given branches with zero weights

; RUN: opt < %s -passes="sample-profile" -sample-profile-file=%S/Inputs/misexpect.prof -pgo-warn-misexpect -S 2>&1  | FileCheck %s

define i32 @main() #0 !dbg !36 {
 ; CHECK-LABEL: @main(
; CHECK-NEXT: for.cond:
; CHECK-NEXT: %0 = load i32, i32* null, align 4, !dbg !44
; CHECK-NEXT: br i1 false, label %for.body, label %for.end, !prof !49
; CHECK: for.body:
; CHECK-NEXT: ret i32 0
; CHECK: for.end:
; CHECK-NEXT: ret i32 0
; NOT: warning:
for.cond:
  %0 = load i32, i32* null, align 4, !dbg !43
  br i1 false, label %for.body, label %for.end, !prof !48

for.body:                                         ; preds = %for.cond
  ret i32 0

for.end:                                          ; preds = %for.cond
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 248211) (llvm/trunk 248217)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "test.cc", directory: "/ssd/llvm_commit")
!2 = !{}
!3 = !{!4}
!4 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"ProfileSummary", !8}
!8 = !{!9, !10, !11, !12, !13, !14, !15, !16, !17, !18}
!9 = !{!"ProfileFormat", !"SampleProfile"}
!10 = !{!"TotalCount", i64 0}
!11 = !{!"MaxCount", i64 0}
!12 = !{!"MaxInternalCount", i64 0}
!13 = !{!"MaxFunctionCount", i64 0}
!14 = !{!"NumCounts", i64 9}
!15 = !{!"NumFunctions", i64 1}
!16 = !{!"IsPartialProfile", i64 0}
!17 = !{!"PartialProfileRatio", double 0.000000e+00}
!18 = !{!"DetailedSummary", !19}
!19 = !{!20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35}
!20 = !{i32 10000, i64 0, i32 0}
!21 = !{i32 100000, i64 0, i32 0}
!22 = !{i32 200000, i64 0, i32 0}
!23 = !{i32 300000, i64 0, i32 0}
!24 = !{i32 400000, i64 0, i32 0}
!25 = !{i32 500000, i64 0, i32 0}
!26 = !{i32 600000, i64 0, i32 0}
!27 = !{i32 700000, i64 0, i32 0}
!28 = !{i32 800000, i64 0, i32 0}
!29 = !{i32 900000, i64 0, i32 0}
!30 = !{i32 950000, i64 0, i32 0}
!31 = !{i32 990000, i64 0, i32 0}
!32 = !{i32 999000, i64 0, i32 0}
!33 = !{i32 999900, i64 0, i32 0}
!34 = !{i32 999990, i64 0, i32 0}
!35 = !{i32 999999, i64 0, i32 0}
!36 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !37, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!37 = !DISubroutineType(types: !38)
!38 = !{!39, !39, !40}
!39 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !41, size: 64, align: 64)
!41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !42, size: 64, align: 64)
!42 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!43 = !DILocation(line: 11, column: 22, scope: !44)
!44 = distinct !DILexicalBlock(scope: !45, file: !1, line: 11, column: 6)
!45 = distinct !DILexicalBlock(scope: !46, file: !1, line: 11, column: 6)
!46 = distinct !DILexicalBlock(scope: !47, file: !1, line: 9, column: 21)
!47 = distinct !DILexicalBlock(scope: !36, file: !1, line: 9, column: 8)
; These branch weights shouldn't be removed by these passes
; CHECK: !{{[0-9]+}} = !{!"branch_weights", i32 0, i32 0}
!48 = !{!"branch_weights", i32 0, i32 0}
