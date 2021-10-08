; XFAIL: -aix
; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck %s
;
; struct A {
;  // Anonymous class exports its symbols into A
;  struct {
;      int y;
;  };
; } a;
;
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_calling_convention	(DW_CC_pass_by_value)
; CHECK-NEXT: DW_AT_name	("A")
;
; CHECK-NOT: NULL
;
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_export_symbols	(true)

%struct.A = type { %struct.anon }
%struct.anon = type { i32 }

@a = global %struct.A zeroinitializer, align 4, !dbg !0

!llvm.module.flags = !{!14, !15}
!llvm.dbg.cu = !{!2}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "simple_anon_class.cpp", directory: "/dir")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS1A")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, scope: !6, file: !3, line: 2, baseType: !9, size: 32)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !6, file: !3, line: 2, size: 32, flags: DIFlagExportSymbols | DIFlagTypePassByValue, elements: !10, identifier: "_ZTSN1AUt_E")
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !9, file: !3, line: 3, baseType: !12, size: 32)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{!"clang version 10.0.0"}
