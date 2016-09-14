; RUN: llc %s -o /dev/null
@0 = internal constant i32 1, !dbg !5

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 139632)", isOptimized: true, emissionKind: FullDebug, file: !8, enums: !2, retainedTypes: !2, globals: !3)
!2 = !{}
!3 = !{!5}
!5 = !DIGlobalVariable(name: "a", line: 2, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7)
!6 = !DIFile(filename: "g.c", directory: "/private/tmp")
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DIFile(filename: "g.c", directory: "/private/tmp")
!9 = !{i32 1, !"Debug Info Version", i32 3}
