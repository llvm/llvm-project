; Make sure Import GUID list for ThinLTO properly set for CSSPGO
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%S/Inputs/csspgo-import-list.prof -S | FileCheck %s
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/csspgo-import-list.prof -o %t.prof
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%t.prof -S | FileCheck %s
; RUN: llvm-profdata show --sample -show-sec-info-only %t.prof | FileCheck %s --check-prefix=CHECK-ORDERED
; RUN: llvm-profdata merge --sample --extbinary --use-md5 %S/Inputs/csspgo-import-list.prof -o %t.md5
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%t.md5 -S | FileCheck %s
; RUN: llvm-profdata show --sample -show-sec-info-only %t.md5 | FileCheck %s --check-prefix=CHECK-ORDERED


declare i32 @_Z5funcBi(i32 %x)
declare i32 @_Z5funcAi(i32 %x)

define dso_local i32 @main() local_unnamed_addr #0 !dbg !18 {
entry:
  br label %for.body, !dbg !25

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add3, !dbg !27

for.body:                                         ; preds = %for.body, %entry
  %x.011 = phi i32 [ 300000, %entry ], [ %dec, %for.body ]
  %r.010 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %call = tail call i32 @_Z5funcBi(i32 %x.011), !dbg !32
  %add = add nuw nsw i32 %x.011, 1, !dbg !31
  %call1 = tail call i32 @_Z5funcAi(i32 %add), !dbg !28
  %add2 = add i32 %call, %r.010, !dbg !34
  %add3 = add i32 %add2, %call1, !dbg !35
  %dec = add nsw i32 %x.011, -1, !dbg !36
  %cmp = icmp eq i32 %x.011, 0, !dbg !38
  br i1 %cmp, label %for.cond.cleanup, label %for.body, !dbg !25
}

; Make sure the ImportGUID stays with entry count metadata for ThinLTO-PreLink
; CHECK: distinct !DISubprogram(name: "main"
; CHECK: !{!"function_entry_count", i64 3, i64 446061515086924981, i64 3815895320998406042, i64 7102633082150537521, i64 -2862076748587597320}

; CHECK-ORDERED: FuncOffsetTableSection {{.*}} {ordered}

attributes #0 = { nofree noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "factor", scope: !2, file: !3, line: 21, type: !13, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !12, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "merged.cpp", directory: "/local/autofdo")
!4 = !{}
!5 = !{!6, !10, !11}
!6 = !DISubprogram(name: "funcA", linkageName: "_Z5funcAi", scope: !3, file: !3, line: 6, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DISubprogram(name: "funcB", linkageName: "_Z5funcBi", scope: !3, file: !3, line: 7, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!11 = !DISubprogram(name: "funcLeaf", linkageName: "_Z8funcLeafi", scope: !3, file: !3, line: 22, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!12 = !{!0}
!13 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !9)
!14 = !{i32 7, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 11.0.0"}
!18 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 11, type: !19, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!9}
!21 = !{!22, !23}
!22 = !DILocalVariable(name: "r", scope: !18, file: !3, line: 12, type: !9)
!23 = !DILocalVariable(name: "x", scope: !24, file: !3, line: 13, type: !9)
!24 = distinct !DILexicalBlock(scope: !18, file: !3, line: 13, column: 3)
!25 = !DILocation(line: 13, column: 3, scope: !26)
!26 = !DILexicalBlockFile(scope: !24, file: !3, discriminator: 2)
!27 = !DILocation(line: 17, column: 3, scope: !18)
!28 = !DILocation(line: 13, column: 10, scope: !29)
!29 = distinct !DILexicalBlock(scope: !30, file: !3, line: 13, column: 37)
!30 = distinct !DILexicalBlock(scope: !24, file: !3, line: 13, column: 3)
!31 = !DILocation(line: 14, column: 29, scope: !29)
!32 = !DILocation(line: 14, column: 21, scope: !33)
!33 = !DILexicalBlockFile(scope: !29, file: !3, discriminator: 2)
!34 = !DILocation(line: 14, column: 19, scope: !29)
!35 = !DILocation(line: 14, column: 7, scope: !29)
!36 = !DILocation(line: 13, column: 33, scope: !37)
!37 = !DILexicalBlockFile(scope: !30, file: !3, discriminator: 6)
!38 = !DILocation(line: 13, column: 26, scope: !39)
!39 = !DILexicalBlockFile(scope: !30, file: !3, discriminator: 2)
