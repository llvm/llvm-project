; RUN: llc %s -mtriple=x86_64-linux-gnu -generate-type-units -o - -filetype=obj \
; RUN: | llvm-dwarfdump -o - - \
; RUN: | FileCheck %s

;; PR51087
;; Check that types that are not referenecd in the CU and have type units
;; do not get unecessary skeleton DIEs in the CU, and that types that have
;; type units but are referenced in the CU still get CU DIEs.
;;
;; In the test (source below):
;; Unused is not used anywhere and should get only a type unit.
;;
;; Outer is used (by global O) so should get a CU DIE, but none of its nested
;; types (Inner, then nested again Enum1 and Enum2) are used so they should not.
;;
;; Ex is not used directly, but its nested type Enum is, so both should get
;; a DIE in the CU. Retained types and enums are emitted after globals, so
;; having Enum used by a local variable lets us check that type DIEs emitted
;; for types that initially only need type units still get a CU DIE later on
;; if required.
;;
;; Generated with `-Xclang -debug-info-kind=unused-types` (for Unused) from:
;; $ cat test.cpp
;; struct Unused {};
;;
;; class Outer {
;; public:
;;   struct Inner {
;;     enum Enum1 { X };
;;     enum Enum2 { Y };
;;     Enum1 one;
;;     Enum2 two;
;;   };
;;
;;   Inner n;
;; } O;
;;
;; struct Ex { enum Enum { X }; };
;; void fun() { Ex::Enum local; }

;; Note: The types without a type_signature match here should only get type
;; units and no CU DIE.
; CHECK: 0x00000000: Type Unit{{.+}} name = 'Outer'{{.+}}  type_signature = [[SIG_Outer:[0-9a-fx]+]]
; CHECK: 0x00000000: Type Unit{{.+}} name = 'Inner'{{.+}}
; CHECK: 0x00000000: Type Unit{{.+}} name = 'Enum1'{{.+}}
; CHECK: 0x00000000: Type Unit{{.+}} name = 'Enum2'{{.+}}
; CHECK: 0x00000000: Type Unit{{.+}} name = 'Enum'{{.+}}   type_signature = [[SIG_Enum:[0-9a-fx]+]]
; CHECK: 0x00000000: Type Unit{{.+}} name = 'Ex'{{.+}}     type_signature = [[SIG_Ex:[0-9a-fx]+]]
; CHECK: 0x00000000: Type Unit{{.+}} name = 'Unused'{{.+}}

;; Check the CU references and skeleton DIEs are emitted correctly.
;; The check-not directives check that Unused doesn't get a DIE in the CU.
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_AT_signature
; CHECK: DW_AT_type ([[DIE_Outer:[0-9a-fx]+]] "Outer")
; CHECK-NOT: DW_AT_signature

;; Outer is referenced in the CU so it needs a DIE, but its nested enums are not
;; and so should not have DIEs here.
; CHECK:      [[DIE_Outer]]:  DW_TAG_class_type
; CHECK-NEXT:    DW_AT_declaration (true)
; CHECK-NEXT:    DW_AT_signature   ([[SIG_Outer]])

; CHECK-NOT: DW_AT_signature
; CHECK: DW_AT_type ([[DIE_Enum:[0-9a-fx]+]] "Ex::Enum")
; CHECK-NOT: DW_AT_signature

;; Ex is not referenced in the CU but its nested type, Enum, is.
; CHECK:      DW_TAG_structure_type
; CHECK-NEXT:     DW_AT_declaration     (true)
; CHECK-NEXT:     DW_AT_signature       ([[SIG_Ex]])
; CHECK-EMPTY:
; CHECK-NEXT:     [[DIE_Enum]]: DW_TAG_enumeration_type
; CHECK-NEXT:         DW_AT_declaration (true)
; CHECK-NEXT:         DW_AT_signature   ([[SIG_Enum]])

;; One last check that Unused has no CU DIE.
; CHECK-NOT: DW_AT_signature

%class.Outer = type { %"struct.Outer::Inner" }
%"struct.Outer::Inner" = type { i32, i32 }

@O = dso_local global %class.Outer zeroinitializer, align 4, !dbg !0

define dso_local void @_Z3funv() !dbg !31 {
entry:
  %local = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %local, metadata !34, metadata !DIExpression()), !dbg !35
  ret void, !dbg !36
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!25, !26, !27, !28, !29}
!llvm.ident = !{!30}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "O", scope: !2, file: !3, line: 13, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !22, globals: !24, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!5, !13, !19}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum1", scope: !6, file: !3, line: 6, baseType: !14, size: 32, elements: !17, identifier: "_ZTSN5Outer5Inner5Enum1E")
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Inner", scope: !7, file: !3, line: 5, size: 64, flags: DIFlagTypePassByValue, elements: !10, identifier: "_ZTSN5Outer5InnerE")
!7 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Outer", file: !3, line: 3, size: 64, flags: DIFlagTypePassByValue, elements: !8, identifier: "_ZTS5Outer")
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !7, file: !3, line: 12, baseType: !6, size: 64, flags: DIFlagPublic)
!10 = !{!11, !12}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "one", scope: !6, file: !3, line: 8, baseType: !5, size: 32)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "two", scope: !6, file: !3, line: 9, baseType: !13, size: 32, offset: 32)
!13 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum2", scope: !6, file: !3, line: 7, baseType: !14, size: 32, elements: !15, identifier: "_ZTSN5Outer5Inner5Enum2E")
!14 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!15 = !{!16}
!16 = !DIEnumerator(name: "Y", value: 0, isUnsigned: true)
!17 = !{!18}
!18 = !DIEnumerator(name: "X", value: 0, isUnsigned: true)
!19 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum", scope: !20, file: !3, line: 15, baseType: !14, size: 32, elements: !17, identifier: "_ZTSN2Ex4EnumE")
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Ex", file: !3, line: 15, size: 8, flags: DIFlagTypePassByValue, elements: !21, identifier: "_ZTS2Ex")
!21 = !{}
!22 = !{!23, !7, !6, !20}
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Unused", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !21, identifier: "_ZTS6Unused")
!24 = !{!0}
!25 = !{i32 7, !"Dwarf Version", i32 5}
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{i32 1, !"wchar_size", i32 4}
!28 = !{i32 7, !"uwtable", i32 1}
!29 = !{i32 7, !"frame-pointer", i32 2}
!30 = !{!"clang version 14.0.0"}
!31 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !3, file: !3, line: 16, type: !32, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !21)
!32 = !DISubroutineType(types: !33)
!33 = !{null}
!34 = !DILocalVariable(name: "local", scope: !31, file: !3, line: 16, type: !19)
!35 = !DILocation(line: 16, column: 23, scope: !31)
!36 = !DILocation(line: 16, column: 30, scope: !31)
