; This checks that .debug_pubnames and .debug_pubtypes can be generated in the DWARF64 format.

; RUN: llc -mtriple=x86_64 -dwarf64 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info -debug-pubnames -debug-pubtypes %t | FileCheck %s

; CHECK:      .debug_info contents:
; CHECK:      0x[[VAR:.+]]:    DW_TAG_variable
; CHECK-NEXT:                    DW_AT_name ("foo")
; CHECK:      0x[[STRUCT:.+]]: DW_TAG_structure_type
; CHECK-NEXT:                    DW_AT_name ("Foo")
; CHECK:      0x[[BASET:.+]]:  DW_TAG_base_type
; CHECK-NEXT:                    DW_AT_name ("int")

; CHECK:      .debug_pubnames contents:
; CHECK-NEXT: length = 0x0000000000000026, format = DWARF64, version = 0x0002, unit_offset =
; CHECK-NEXT: Offset     Name
; CHECK-NEXT: 0x00000000[[VAR]] "foo"

; CHECK:      .debug_pubtypes contents:
; CHECK-NEXT: length = 0x0000000000000032, format = DWARF64, version = 0x0002, unit_offset =
; CHECK-NEXT: Offset     Name
; CHECK-NEXT: 0x00000000[[STRUCT]] "Foo"
; CHECK-NEXT: 0x00000000[[BASET]] "int"

; IR generated and reduced from:
; $ cat foo.c
; struct Foo { int bar; };
; struct Foo foo;
; $ clang -g -gpubnames -S -emit-llvm foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

%struct.Foo = type { i32 }

@foo = dso_local global %struct.Foo zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false)
!3 = !DIFile(filename: "foo.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !3, line: 1, size: 32, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "bar", scope: !6, file: !3, line: 1, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 12.0.0"}
