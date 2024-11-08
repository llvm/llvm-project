; RUN: opt -passes=dce -S %s -o - | FileCheck %s

;; Check that DCE doesn't remove a dbg intrinsic that has an empty metadata
;; operand.
; CHECK: #dbg_declare(![[EMPTY:[0-9]+]],
; CHECK: ![[EMPTY]] = !{}

define dso_local void @fun() local_unnamed_addr #0 !dbg !9 {
entry:
  call void @llvm.dbg.declare(metadata !{}, metadata !13, metadata !DIExpression()), !dbg !15
  ret void, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 16.0.0"}
!9 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{!13}
!13 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 1, type: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 1, column: 18, scope: !9)
!16 = !DILocation(line: 1, column: 21, scope: !9)

