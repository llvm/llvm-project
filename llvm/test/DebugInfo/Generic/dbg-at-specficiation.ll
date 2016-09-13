; RUN: llc  < %s | FileCheck %s
; Radar 10147769
; Do not unnecessarily use AT_specification DIE.
; CHECK-NOT: AT_specification

@a = common global [10 x i32] zeroinitializer, align 16, !dbg !5

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 140253)", isOptimized: true, emissionKind: FullDebug, file: !11, enums: !2, retainedTypes: !2, globals: !3)
!2 = !{}
!3 = !{!5}
!5 = !DIGlobalVariable(name: "a", line: 1, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7)
!6 = !DIFile(filename: "x.c", directory: "/private/tmp")
!7 = !DICompositeType(tag: DW_TAG_array_type, size: 320, align: 32, baseType: !8, elements: !9)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DISubrange(count: 10)
!11 = !DIFile(filename: "x.c", directory: "/private/tmp")
!12 = !{i32 1, !"Debug Info Version", i32 3}
