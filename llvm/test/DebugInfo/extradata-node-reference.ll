;; Test verifies that node reference in the extraData field are handled correctly
;; when used with tags like DW_TAG_member, DW_TAG_inheritance etc.

; REQUIRES: object-emission
; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s -check-prefix=CHECK-IR
; RUN: verify-uselistorder %s

; Example 1: BitField with storage offset (extraData: i64 0)
%struct.BitField = type { i8 }
@bf = global %struct.BitField zeroinitializer, !dbg !9

; Example 2: Static member with constant value (extraData: i32 42)
%struct.Static = type { i32 }
@st = global %struct.Static zeroinitializer, !dbg !16

; Example 3: Discriminant value for variant (extraData: i32 100)
%union.Variant = type { [8 x i8] }
@var = global %union.Variant zeroinitializer, !dbg !24

; Example 4: Inheritance VBPtr offset (extraData: i32 0)
%class.Derived = type { i32 }
@der = global %class.Derived zeroinitializer, !dbg !35

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !8)
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{!9, !16, !24, !35}

; extraData node definitions
!15 = !{i64 0}       ; BitField storage offset
!22 = !{i32 42}      ; Static member constant value
!33 = !{i32 100}     ; Discriminant value
!41 = !{i32 0}       ; VBPtr offset

; CHECK-IR: !9 = !DIDerivedType(tag: DW_TAG_member, name: "const_val", scope: !7, file: !3, line: 11, baseType: !10, flags: DIFlagStaticMember, extraData: !12)
; CHECK-IR: !12 = !{i32 42}
; CHECK-IR: !20 = !DIDerivedType(tag: DW_TAG_member, name: "variant_some", scope: !17, file: !3, baseType: !11, size: 32, extraData: !21)
; CHECK-IR: !21 = !{i32 100}
; CHECK-IR: !27 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !25, baseType: !28, extraData: !29)
; CHECK-IR: !29 = !{i32 0}
; CHECK-IR: !32 = !DIDerivedType(tag: DW_TAG_member, name: "field", scope: !30, file: !3, line: 6, baseType: !11, size: 3, flags: DIFlagBitField, extraData: !33)
; CHECK-IR: !33 = !{i64 0}

; CHECK: {{.*}} DW_TAG_variable
; CHECK: {{.*}} DW_AT_name	("bf")
; CHECK: {{.*}} DW_TAG_member
; CHECK: {{.*}} DW_AT_name	("field")
; === BitField: extraData holds storage offset ===
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "bf", scope: !0, file: !1, line: 5, type: !11, isLocal: false, isDefinition: true)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "BitField", file: !1, line: 5, size: 8, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "field", scope: !11, file: !1, line: 6, baseType: !14, size: 3, flags: DIFlagBitField, extraData: !15)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; CHECK: {{.*}} DW_TAG_variable
; CHECK: {{.*}} DW_AT_name	("st")
; CHECK: {{.*}} DW_TAG_member
; CHECK: {{.*}} DW_AT_name	("const_val")
; CHECK: {{.*}} DW_AT_const_value	(42)
; === Static Member: extraData holds constant value ===
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "st", scope: !0, file: !1, line: 10, type: !18, isLocal: false, isDefinition: true)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Static", file: !1, line: 10, size: 32, elements: !19)
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "const_val", scope: !18, file: !1, line: 11, baseType: !21, flags: DIFlagStaticMember, extraData: !22)
!21 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)

; CHECK: {{.*}} DW_TAG_variable
; CHECK: {{.*}} DW_AT_name	("var")
; CHECK: {{.*}} DW_TAG_member
; CHECK: {{.*}} DW_AT_name	("variant_none")
; CHECK: {{.*}} DW_AT_discr_value	(0x64)
; === Discriminant: extraData holds discriminant value ===
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression())
!25 = distinct !DIGlobalVariable(name: "var", scope: !0, file: !1, line: 15, type: !26, isLocal: false, isDefinition: true)
!26 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Variant", file: !1, line: 15, size: 128, elements: !27)
!27 = !{!28}
!28 = !DICompositeType(tag: DW_TAG_variant_part, scope: !26, file: !1, size: 128, elements: !29, discriminator: !30)
!29 = !{!31, !32}
!30 = !DIDerivedType(tag: DW_TAG_member, scope: !28, file: !1, baseType: !14, size: 32, align: 32, flags: DIFlagArtificial)
!31 = !DIDerivedType(tag: DW_TAG_member, name: "variant_none", scope: !28, file: !1, baseType: !14, size: 32)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "variant_some", scope: !28, file: !1, baseType: !14, size: 32, extraData: !33)

; CHECK: {{.*}} DW_TAG_variable
; CHECK: {{.*}} DW_AT_name	("der")
; CHECK: {{.*}} DW_TAG_inheritance
; CHECK: {{.*}} DW_AT_type	({{.*}} "Base")
; === Inheritance: extraData holds VBPtr offset ===
!35 = !DIGlobalVariableExpression(var: !36, expr: !DIExpression())
!36 = distinct !DIGlobalVariable(name: "der", scope: !0, file: !1, line: 20, type: !37, isLocal: false, isDefinition: true)
!37 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Derived", file: !1, line: 20, size: 32, elements: !38)
!38 = !{!39}
!39 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !37, baseType: !40, extraData: !41)
!40 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Base", file: !1, line: 19, size: 32)
