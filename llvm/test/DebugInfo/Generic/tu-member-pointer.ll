; RUN: %llc_dwarf -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE:0x[0-9a-f]+]]}
; CHECK: [[TYPE]]:   DW_TAG_base_type
; IR generated from clang -g with the following source:
; struct Foo {
;   int e;
; };
; int Foo:*x = 0;

source_filename = "test/DebugInfo/Generic/tu-member-pointer.ll"

@x = global i64 -1, align 8, !dbg !0

!llvm.dbg.cu = !{!6}
!llvm.module.flags = !{!10, !11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: null, file: !2, line: 4, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "foo.cpp", directory: ".")
!3 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !4, extraData: !5)
!4 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !2, line: 1, flags: DIFlagFwdDecl, identifier: "_ZTS3Foo")
!6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.4", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, retainedTypes: !8, globals: !9, imports: !7)
!7 = !{}
!8 = !{!5}
!9 = !{!0}
!10 = !{i32 2, !"Dwarf Version", i32 2}
!11 = !{i32 1, !"Debug Info Version", i32 3}

