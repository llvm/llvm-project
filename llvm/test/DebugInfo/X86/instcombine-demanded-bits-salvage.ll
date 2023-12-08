; RUN: opt -mtriple=x86_64-- %s -S -passes=instcombine  -o - | FileCheck %s
; Verify that demanded bits optimisations don't affect debuginfo
; variable values.
; Bugzilla #44371

@a = common dso_local local_unnamed_addr global i32 0, align 4

define dso_local i32 @p() local_unnamed_addr !dbg !11 {
entry:
  %conv = load i32, ptr @a, align 4, !dbg !14
  %0 = and i32 %conv, 65535, !dbg !14
  ; CHECK: metadata !DIExpression(DW_OP_constu, 65535, DW_OP_and, DW_OP_stack_value))
  call void @llvm.dbg.value(metadata i32 %0, metadata !15, metadata !DIExpression()), !dbg !14
  %1 = lshr i32 %0, 12, !dbg !14
  %2 = and i32 %1, 8, !dbg !14
  %3 = xor i32 %2, 8, !dbg !14
  ret i32 0, !dbg !14
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "instcomnine.ll", directory: "/temp/bz44371")
!2 = !{}
!3 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!""}
!11 = distinct !DISubprogram(name: "p", scope: !1, file: !1, line: 6, type: !12, scopeLine: 6, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{!7}
!14 = !DILocation(line: 4, column: 35, scope: !11)
!15 = !DILocalVariable(name: "p_28", arg: 1, scope: !11, file: !1, line: 16, type: !7)
