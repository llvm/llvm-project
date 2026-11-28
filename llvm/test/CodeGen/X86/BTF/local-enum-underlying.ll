; RUN: split-file %s %t
; RUN: llc -mtriple=x86_64-linux-gnu %t/typedef.ll -o - | FileCheck %s --check-prefix=TYPEDEF
; RUN: llc -mtriple=x86_64-linux-gnu %t/wide.ll -o - | FileCheck %s --check-prefix=WIDE --implicit-check-not=BTF_KIND_ENUM

; TYPEDEF: BTF_KIND_ENUM(
; TYPEDEF-NEXT: .long 100663297
; TYPEDEF-NEXT: .long 2
; TYPEDEF: .ascii "value"

; WIDE: .section .BTF
; WIDE: BTF_KIND_FUNC(

;--- typedef.ll
define void @f() !dbg !4 { ret void }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "enum.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"BTF", i32 1}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !DISubroutineType(types: !{null}), unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !{!5})
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", scope: !4, baseType: !6, size: 16, elements: !{!DIEnumerator(name: "value", value: 42, isUnsigned: true)})
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "u16", baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)

;--- wide.ll
define void @f() !dbg !4 { ret void }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "enum.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"BTF", i32 1}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !DISubroutineType(types: !{null}), unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !{!5})
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", scope: !4, baseType: !6, size: 128, elements: !{!DIEnumerator(name: "value", value: 42, isUnsigned: true)})
!6 = !DIBasicType(name: "unsigned __int128", size: 128, encoding: DW_ATE_unsigned)
