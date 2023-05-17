; RUN: llvm-reduce %s -o %t --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --input-file

; CHECK: @foo

define void @foo() !dbg !7 {
entry:
  %a = alloca i32
  call void @llvm.dbg.declare(metadata ptr %a, metadata !13, metadata !DIExpression()), !dbg !14
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/a.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = distinct !DISubprogram(name: "main", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!8 = !DIFile(filename: "/tmp/a.c", directory: "")
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{}
!13 = !DILocalVariable(name: "a", scope: !7, file: !8, line: 2, type: !11)
!14 = !DILocation(line: 2, column: 7, scope: !7)
