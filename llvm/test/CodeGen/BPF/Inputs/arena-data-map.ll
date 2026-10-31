
%struct.anon = type { ptr, ptr, ptr }

@my_arena = dso_local global %struct.anon zeroinitializer, section ".maps", align 8, !dbg !0
@llvm.compiler.used = appending global [1 x ptr] [ptr @my_arena], section "llvm.metadata"

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!24, !25, !26}
!llvm.errno.tbaa = !{!28}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "my_arena", scope: !2, file: !5, line: 7, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "m.c", directory: "/tmp")
!4 = !{!0}
!5 = !DIFile(filename: "m.c", directory: "/tmp")
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !5, line: 3, size: 192, elements: !7)
!7 = !{!8, !14, !19}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "type", scope: !6, file: !5, line: 4, baseType: !9, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 1056, elements: !12)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DISubrange(count: 33)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "map_flags", scope: !6, file: !5, line: 5, baseType: !15, size: 64, offset: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 32768, elements: !17)
!17 = !{!18}
!18 = !DISubrange(count: 1024)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "max_entries", scope: !6, file: !5, line: 6, baseType: !20, size: 64, offset: 128)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 512, elements: !22)
!22 = !{!23}
!23 = !DISubrange(count: 16)
!24 = !{i32 7, !"Dwarf Version", i32 5}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = !{i32 7, !"frame-pointer", i32 2}
!28 = !{!29, !30, i64 0}
!29 = !{!"__libc_errno", !30, i64 0}
!30 = !{!"int", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
