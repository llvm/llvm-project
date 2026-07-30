; RUN: llc  < %s | FileCheck %s
; Radar 10147769
; Do not unnecessarily use AT_specification DIE.
; CHECK-NOT: AT_specification

source_filename = "test/DebugInfo/Generic/dbg-at-specficiation.ll"

@a = common global [10 x i32] zeroinitializer, align 16, !dbg !0

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "x.c", directory: "/private/tmp")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 320, align: 32, elements: !5)
!4 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!5 = !{!6}
!6 = !DISubrange(count: 10)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.0 (trunk 140253)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, retainedTypes: !8, globals: !9)
!8 = !{}
!9 = !{!0}
!10 = !{i32 1, !"Debug Info Version", i32 3}
