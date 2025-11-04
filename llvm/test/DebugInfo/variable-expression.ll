; RUN: llvm-as < %s | llvm-dis | llc -mtriple=x86_64 -O0 -filetype=obj -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

; A test to verify that DIVariableExpression emits
; DW_OP_GNU_variable_value.

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name [DW_FORM_strp] {{.*"pck__my_rec".*}}
; CHECK-NEXT: DW_AT_bit_size {{.*DW_OP_GNU_variable_value}}

; ModuleID = 'foo.adb'
source_filename = "foo.adb"

@pck__my_length_LAST = external dso_local global i32, align 4

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = distinct !DICompileUnit(language: DW_LANG_Ada95, file: !3, producer: "GNAT/LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, imports: !9, splitDebugInlining: false, retainedTypes: !17)
!3 = !DIFile(filename: "foo.adb", directory: "")
!4 = !{!5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "pck__my_length_LAST", scope: !2, file: !7, line: 23, type: !8, isLocal: false, isDefinition: false, align: 32)
!7 = !DIFile(filename: "pck.ads", directory: "")
!8 = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !2, entity: !11, file: !3, line: 16)
!11 = !DIModule(scope: null, name: "pck")
!12 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 18, type: !13, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !{}
!16 = !DILocation(line: 19, column: 4, scope: !12)
!17 = !{!18}
!18 = !DICompositeType(tag: DW_TAG_structure_type, name: "pck__my_rec", file: !7, line: 26, size: !19, align: 32, elements: !21, identifier: "pck__my_rec")
!19 = !DIVariableExpression(expr: !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 0, DW_OP_over, DW_OP_swap, DW_OP_over, DW_OP_over, DW_OP_lt, DW_OP_neg, DW_OP_rot, DW_OP_xor, DW_OP_and, DW_OP_xor, DW_OP_constu, 32, DW_OP_mul, DW_OP_constu, 32, DW_OP_plus), vars: !20)
!20 = !{!6}
!21 = !{!22, !23}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "x", file: !7, line: 27, baseType: !8, size: 32, align: 32)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "a", file: !7, line: 28, baseType: !24, offset: 32)
!24 = !DICompositeType(tag: DW_TAG_array_type, name: "pck__my_array", file: !7, line: 24, baseType: !25, align: 32, elements: !27)
!25 = !DISubrangeType(name: "natural", file: !26, line: 1, size: 32, align: 32, baseType: !8, lowerBound: i64 0, upperBound: i64 2147483647)
!26 = !DIFile(filename: "system.ads", directory: "")
!27 = !{!28}
!28 = !DISubrangeType(baseType: !8, lowerBound: i64 1, upperBound: !6)
