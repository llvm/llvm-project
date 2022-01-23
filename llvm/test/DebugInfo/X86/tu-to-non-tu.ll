; RUN: llc -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -debug-info -debug-types - | FileCheck %s

; Test that a type unit referencing a non-type unit produces a declaration of
; the referent in the referee.

; Also check that an attempt to reference an internal linkage (defined in an anonymous
; namespace) type from a type unit (could happen with a pimpl idiom, for instance -
; it does mean the linkage-having type can only be defined in one translation
; unit anyway) forces the referent to not be placed in a type unit (because the
; declaration of the internal linkage type would be ambiguous/wouldn't allow a
; consumer to find the definition with certainty)

; CHECK: Type Unit:

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"t1"

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_name {{.*}}"t2"

; CHECK: Compile Unit:

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"t2"
; CHECK-NEXT: DW_AT_byte_size

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"t3"

; CHECK: DW_TAG_namespace
; CHECK-NOT: {{DW_TAG|DW_AT}}

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"t4"

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !13}
!llvm.ident = !{!12}

!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 5.0.0 (trunk 294954) (llvm/trunk 294959)", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "tu-to-non-tu.dwo", emissionKind: FullDebug, enums: !4, retainedTypes: !14)
!3 = !DIFile(filename: "tu.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!4 = !{}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !3, line: 5, size: 8, elements: !7, identifier: "_ZTS2t1")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !6, file: !3, line: 6, baseType: !9, size: 8)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: !3, line: 2, size: 8, elements: !4)
!10 = !DINamespace(scope: null)
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 5.0.0 (trunk 294954) (llvm/trunk 294959)"}
!13 = !{i32 2, !"Dwarf Version", i32 5}
!14 = !{!6, !15}
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t3", file: !3, line: 5, size: 8, elements: !16, identifier: "_ZTS2t3")
!16 = !{!17}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !15, file: !3, line: 6, baseType: !18, size: 8)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t4", scope: !10, file: !3, line: 2, size: 8, elements: !4)
