; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj %s -o %t
; RUN: llvm-objcopy --dump-section .BTF=- %t | %python %p/../../BPF/BTF/print_btf.py - | FileCheck %s
; CHECK: STRUCT 'Node' size=4 vlen=1
; CHECK-NEXT: 'value' type_id={{[0-9]+}} bits_offset=0
; CHECK-NOT: singleton

@n = global i32 0, !dbg !9
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, emissionKind: FullDebug, globals: !{!9})
!1 = !DIFile(filename: "static.cc", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"BTF", i32 1}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Node", file: !1, size: 32, elements: !{!6, !7})
!6 = !DIDerivedType(tag: DW_TAG_member, name: "singleton", scope: !5, baseType: !5, flags: DIFlagStaticMember)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !5, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "n", scope: !0, file: !1, type: !5, isLocal: false, isDefinition: true)
