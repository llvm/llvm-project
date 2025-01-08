; RUN: opt %s -passes=declare-to-assign -S | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators %s -passes=declare-to-assign -S | FileCheck %s

;; Check AssignmentTrackingPass ignores a dbg.declare with an empty metadata
;; location operand.

; CHECK:      #dbg_declare
; CHECK-NOT:  #dbg_assign

define dso_local void @_Z3funv() #0 !dbg !10 {
entry:
  call void @llvm.dbg.declare(metadata !13, metadata !14, metadata !DIExpression()), !dbg !16
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 17.0.0"}
!10 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !{}
!14 = !DILocalVariable(name: "x", scope: !10, file: !1, line: 1, type: !15)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DILocation(line: 1, column: 18, scope: !10)
