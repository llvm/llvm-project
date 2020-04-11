; RUN: llc -O0 -fast-isel=1 -o - -print-after="finalize-isel" %s 2>&1 | FileCheck %s

; Check that we emit a DBG_VALUE for the `@llvm.dbg.value` which has `undef` has first arg.

target triple = "arm64-apple-ios13.4.0"
define void @foo() !dbg !6 {
  ; CHECK: DBG_VALUE $noreg, $noreg, !"1", !DIExpression()
  call void @llvm.dbg.value(metadata i32* undef, metadata !9, metadata !DIExpression()), !dbg !11
  ret void, !dbg !12
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.ll", directory: "/")
!2 = !{}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !6)
!12 = !DILocation(line: 2, column: 1, scope: !6)
