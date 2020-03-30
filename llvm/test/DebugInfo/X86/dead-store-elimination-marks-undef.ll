; RUN: opt -mtriple=x86_64-- -S --dse %s  -o - | FileCheck %s
; Ensure that we can mark a value as undefined when performing dead 
; store elimination.
; Bugzilla #45080

@b = common dso_local local_unnamed_addr global i32 0, align 1

define dso_local i32 @main() local_unnamed_addr !dbg !7 {
  %1 = alloca i32, align 4
  %2 = load i32, i32* @b, align 1, !dbg !13
  ; CHECK: call void @llvm.dbg.value(metadata i32 undef
  call void @llvm.dbg.value(metadata i32 %2, metadata !12, metadata !DIExpression()), !dbg !13
  store i32 %2, i32* %1, align 4, !dbg !13
  ret i32 0, !dbg !13
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2, nameTableKind: None)
!1 = !DIFile(filename: "dead-store-elimination-marks-undef.ll", directory: "/temp/bz45080")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!"clang version 10.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "l_2864", scope: !7, file: !1, line: 4, type: !10)
!13 = !DILocation(line: 5, column: 12, scope: !7)

