; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj %s -o %t
; RUN: llvm-objcopy --dump-section .BTF=- %t | %python %p/../../BPF/BTF/print_btf.py - | FileCheck %s

; Deferred pointees are emitted in first-use order, not by metadata address.
; CHECK: FWD 'berry'
; CHECK-NEXT: {{.*}} FWD 'apple'

define void @f() !dbg !4 { ret void }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "order.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"BTF", i32 1}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !DISubroutineType(types: !{null}), unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !{!5})
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "local", scope: !4, size: 128, elements: !{!6, !7})
!6 = !DIDerivedType(tag: DW_TAG_member, name: "pa", scope: !5, baseType: !8, size: 64, offset: 64)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "pb", scope: !5, baseType: !9, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "apple", file: !1, size: 32)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "berry", file: !1, size: 32)
