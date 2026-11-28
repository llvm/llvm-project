; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj %s -o %t
; RUN: llvm-objcopy --dump-section .BTF=- %t | %python %p/../../BPF/BTF/print_btf.py - | FileCheck %s
; CHECK: STRUCT 'holder' size=8 vlen=1
; CHECK-NEXT: 'p' type_id=[[PTR:[0-9]+]] bits_offset=0
; CHECK: {{\[}}[[PTR]]] PTR '(anon)' type_id=[[FWD:[0-9]+]]
; CHECK: {{\[}}[[FWD]]] FWD 'payload' fwd_kind=struct

@h = global ptr null, !dbg !0
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!3, !4}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "h", scope: !2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !6, emissionKind: FullDebug, globals: !{!0})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"BTF", i32 1}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "holder", size: 64, elements: !{!7})
!6 = !DIFile(filename: "test.c", directory: "/")
!7 = !DIDerivedType(tag: DW_TAG_member, name: "p", scope: !5, baseType: !8, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "payload", size: 256, elements: !{!10})
!10 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !9, baseType: !11, size: 256)
!11 = !DIBasicType(name: "complex", size: 256, encoding: DW_ATE_complex_float)
