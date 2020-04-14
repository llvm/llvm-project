; UNSUPPORTED: system-windows
;
; This test verifies that we do the right thing with DIFlagExportSymbols which is the new
; behaviour and without the DIFlagExportSymbols which is the old behavior for the given
; definitions below.
;
;```
; struct A{
;   struct {
;    int x;
;   };
;   struct {
;    int y;
;   };
;   struct {
;    int z;
;   } unnamed;
; } a;
;```
;
; RUN: %clangxx_host -g -c -o %t.o %s
; RUN: lldb-test symbols -dump-clang-ast %t.o | FileCheck %s
; RUN: llvm-dwarfdump %t.o | FileCheck %s --check-prefix DWARFDUMP

%struct.A = type { %struct.anon, %struct.anon.0, %struct.anon.1 }
%struct.anon = type { i32 }
%struct.anon.0 = type { i32 }
%struct.anon.1 = type { i32 }

@a = global %struct.A zeroinitializer, align 4, !dbg !0

!llvm.module.flags = !{!21, !22}
!llvm.dbg.cu = !{!2}
!llvm.ident = !{!23}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 11, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "anon_old_new.cpp", directory: "/dir")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 96, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS1A")
; CHECK: struct A definition
!7 = !{!8, !13, !17}
!8 = !DIDerivedType(tag: DW_TAG_member, scope: !6, file: !3, line: 2, baseType: !9, size: 32)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !6, file: !3, line: 2, size: 32, flags: DIFlagExportSymbols | DIFlagTypePassByValue, elements: !10, identifier: "_ZTSN1AUt_E")
; Correctly identify an anonymous class with DIFlagExportSymbols
; CHECK: struct definition
; CHECK: DefinitionData is_anonymous pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !9, file: !3, line: 3, baseType: !12, size: 32)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_member, scope: !6, file: !3, line: 5, baseType: !14, size: 32, offset: 32)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !6, file: !3, line: 5, size: 32, flags: DIFlagTypePassByValue, elements: !15, identifier: "_ZTSN1AUt0_E")
; Correctly identify an anonymous class without DIFlagExportSymbols
; This works b/c we have additional checks when we fields to A.
; CHECK: struct definition
; CHECK: DefinitionData is_anonymous pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !14, file: !3, line: 6, baseType: !12, size: 32)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "unnamed", scope: !6, file: !3, line: 10, baseType: !18, size: 32, offset: 64)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !6, file: !3, line: 8, size: 32, flags: DIFlagTypePassByValue, elements: !19, identifier: "_ZTSN1AUt1_E")
; Correctly identify an unamed class
; CHECK: struct definition
; CHECK: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !18, file: !3, line: 9, baseType: !12, size: 32)
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{!"clang version 10.0.0"}

; DWARFDUMP: DW_TAG_structure_type
; DWARFDUMP: DW_AT_name	("A")
;
; DWARFDUMP: DW_TAG_structure_type
; DWARFDUMP: DW_AT_export_symbols	(true)
;
; DWARFDUMP: DW_TAG_structure_type
; DWARFDUMP-NOT: DW_AT_export_symbols   (true)
