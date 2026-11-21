; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj %s -o %t
; RUN: llvm-objcopy --dump-section .BTF=- %t | %python %p/../../BPF/BTF/print_btf.py - | FileCheck %s

; A pointer to a type from the first CU must not resolve to a same-named
; definition from the second CU. If its definition is not emitted, use a FWD.
; CHECK: STRUCT 'container' size=8 vlen=1
; CHECK-NEXT: 'p' type_id=[[PTR:[0-9]+]] bits_offset=0
; CHECK: {{\[}}[[PTR]]] PTR '(anon)' type_id=[[FWD:[0-9]+]]
; CHECK: STRUCT 'payload' size=8 vlen=1
; CHECK: {{\[}}[[FWD]]] FWD 'payload' fwd_kind=struct

define void @first() !dbg !4 { ret void }
define void @second() !dbg !5 { ret void }

!llvm.dbg.cu = !{!0, !1}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !6, emissionKind: FullDebug)
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !7, emissionKind: FullDebug)
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"BTF", i32 1}
!4 = distinct !DISubprogram(name: "first", scope: !6, file: !6, type: !8, unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !10)
!5 = distinct !DISubprogram(name: "second", scope: !7, file: !7, type: !8, unit: !1, spFlags: DISPFlagDefinition, retainedNodes: !11)
!6 = !DIFile(filename: "a.c", directory: "/")
!7 = !DIFile(filename: "b.c", directory: "/")
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!12}
!11 = !{!13}
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "container", scope: !4, file: !6, size: 64, elements: !14)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "payload", scope: !5, file: !7, size: 64, elements: !18)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "p", scope: !12, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "payload", file: !6, size: 32)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !13, baseType: !20, size: 64)
!20 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
