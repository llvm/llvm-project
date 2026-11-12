; RUN: llc -mtriple=x86_64 -generate-type-units -dwarf-version=5 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-names %t | FileCheck %s

; Two structs that have different names but are structurally identical:
;    namespace MyNamespace {
;      struct MyStruct1 {
;          char c1;
;      };
;      struct MyStruct2 {
;          char c2;
;      };
;    } // namespace MyNamespace
;    MyNamespace::MyStruct1 gv1;
;    MyNamespace::MyStruct2 gv2;

; Using two TUs, this should produce DIE structures with the same offset.
; We test that accelerator table generation works with this.

; CHECK:      String: {{.*}} "MyStruct2"
; CHECK-NEXT:      Entry @ {{.*}} {
; CHECK-NEXT:        Abbrev: [[ABBREV:0x.*]]
; CHECK-NEXT:        Tag: DW_TAG_structure_type
; CHECK-NEXT:        DW_IDX_type_unit: 0x01
; CHECK-NEXT:        DW_IDX_die_offset: [[DieOffsetStruct:0x.*]]
; CHECK-NEXT:        DW_IDX_parent: Entry @ [[Parent2:0x.*]]
; CHECK:      String: {{.*}} "MyNamespace"
; CHECK-NEXT:      Entry @ [[Parent1:0x.*]] {
; CHECK-NEXT:        Abbrev: {{.*}}
; CHECK-NEXT:        Tag: DW_TAG_namespace
; CHECK-NEXT:        DW_IDX_type_unit: 0x00
; CHECK-NEXT:        DW_IDX_die_offset: [[DieOffsetNamespace:0x.*]]
; CHECK-NEXT:        DW_IDX_parent: <parent not indexed>
; CHECK-NEXT:      }
; CHECK-NEXT:      Entry @ [[Parent2]] {
; CHECK-NEXT:        Abbrev: {{.*}}
; CHECK-NEXT:        Tag: DW_TAG_namespace
; CHECK-NEXT:        DW_IDX_type_unit: 0x01
; CHECK-NEXT:        DW_IDX_die_offset: [[DieOffsetNamespace:0x.*]]
; CHECK-NEXT:        DW_IDX_parent: <parent not indexed>
; CHECK-NEXT:      }
; CHECK:      String: {{.*}} "MyStruct1"
; CHECK-NEXT:      Entry @ {{.*}} {
; CHECK-NEXT:        Abbrev: [[ABBREV]]
; CHECK-NEXT:        Tag: DW_TAG_structure_type
; CHECK-NEXT:        DW_IDX_type_unit: 0x00
; CHECK-NEXT:        DW_IDX_die_offset: [[DieOffsetStruct:0x.*]]
; CHECK-NEXT:        DW_IDX_parent: Entry @ [[Parent1]]


%"struct.MyNamespace::MyStruct1" = type { i8 }
%"struct.MyNamespace::MyStruct2" = type { i8 }

@gv1 = dso_local global %"struct.MyNamespace::MyStruct1" zeroinitializer, align 1, !dbg !0
@gv2 = dso_local global %"struct.MyNamespace::MyStruct2" zeroinitializer, align 1, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17, !18, !19}
!llvm.ident = !{!20}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "gv1", scope: !2, file: !3, line: 10, type: !12, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "blah", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false)
!3 = !DIFile(filename: "two_tus.cpp", directory: "blah", checksumkind: CSK_MD5, checksum: "69acf04f32811fe7fd35449b58c3f5b1")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "gv2", scope: !2, file: !3, line: 11, type: !7, isLocal: false, isDefinition: true)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct2", scope: !8, file: !3, line: 5, size: 8, flags: DIFlagTypePassByValue, elements: !9, identifier: "_ZTSN11MyNamespace9MyStruct2E")
!8 = !DINamespace(name: "MyNamespace", scope: null)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "c2", scope: !7, file: !3, line: 6, baseType: !11, size: 8)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct1", scope: !8, file: !3, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !13, identifier: "_ZTSN11MyNamespace9MyStruct1E")
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "c1", scope: !12, file: !3, line: 3, baseType: !11, size: 8)
!15 = !{i32 7, !"Dwarf Version", i32 5}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 7, !"uwtable", i32 2}
!19 = !{i32 7, !"frame-pointer", i32 2}
!20 = !{!"blah"}
