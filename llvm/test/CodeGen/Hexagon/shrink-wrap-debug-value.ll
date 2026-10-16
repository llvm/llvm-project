; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: shrink_wrap_debug:
; CHECK-COUNT-2: r29 = add(r29
; CHECK-NOT: r29 = add(r29

define void @shrink_wrap_debug() !dbg !4 {
entry:
  %object = alloca i32, align 4
  %0 = cmpxchg weak ptr %object, i32 0, i32 1 seq_cst seq_cst, align 4
    #dbg_value(ptr %object, !7, !DIExpression(), !8)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "shrink_wrap_debug", scope: !1, file: !1, line: 1, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !6)
!5 = !DISubroutineType(types: !2)
!6 = !{!7}
!7 = !DILocalVariable(name: "object", scope: !4, file: !1, line: 1, type: !9)
!8 = !DILocation(line: 1, column: 1, scope: !4)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
