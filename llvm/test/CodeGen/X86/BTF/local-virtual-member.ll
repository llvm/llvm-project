; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj %s -o %t
; RUN: llvm-objcopy --dump-section .BTF=- %t | %python %p/../../BPF/BTF/print_btf.py - | FileCheck %s --implicit-check-not="STRUCT 'Wrapper'" --implicit-check-not="STRUCT 'V'" --implicit-check-not="ARRAY"

; A virtual-base record cannot appear by value, even inside an array and
; behind a typedef and qualifier. Pointers to the wrapper still work.
; CHECK: STRUCT 'Holder' size=8 vlen=1
; CHECK-NEXT: 'p' type_id=[[PTR:[0-9]+]] bits_offset=0
; CHECK-NEXT: {{\[}}[[PTR]]] PTR '(anon)' type_id=[[FWD:[0-9]+]]
; CHECK: {{\[}}[[FWD]]] FWD 'Wrapper' fwd_kind=struct

define void @f() !dbg !4 { ret void }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "virtual.cc", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"BTF", i32 1}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !DISubroutineType(types: !{null}), unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !{!5, !8})
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Holder", scope: !4, size: 64, elements: !{!6})
!6 = !DIDerivedType(tag: DW_TAG_member, name: "p", scope: !5, baseType: !7, size: 64)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Wrapper", scope: !4, size: 256, elements: !{!9})
!9 = !DIDerivedType(tag: DW_TAG_member, name: "items", scope: !8, baseType: !10, size: 256)
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "Items", baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !12)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 256, elements: !{!DISubrange(count: 2)})
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "V", file: !1, size: 128, elements: !{!14})
!14 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !13, baseType: !15, flags: DIFlagVirtual)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !1, size: 32)
