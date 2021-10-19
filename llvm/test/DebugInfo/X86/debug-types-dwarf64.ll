; This checks that .debug_types can be generated in the DWARF64 format.

; RUN: llc -mtriple=x86_64 -dwarf-version=4 -dwarf64 -generate-type-units -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-types -v %t | FileCheck %s

; CHECK:      .debug_types contents:
; CHECK-NEXT: Type Unit: {{.+}}, format = DWARF64, {{.+}}, type_offset = 0x[[OFF:.+]] (next unit at

; CHECK:      0x00000027:     DW_TAG_type_unit

; CHECK:      0x0000[[OFF]]:    DW_TAG_structure_type
; CHECK-NEXT:                     DW_AT_calling_convention
; CHECK-NEXT:                     DW_AT_name [DW_FORM_strp] ({{.+}} = "Foo")

; CHECK:      0x{{.+}}:           DW_TAG_member
; CHECK-NEXT:                       DW_AT_name [DW_FORM_strp] ({{.+}} = "bar")
; CHECK-NEXT:                       DW_AT_type [DW_FORM_ref4] (cu + 0x[[BTOFF:.+]] => {0x0000[[BTOFF]]} "int")

; CHECK:      0x{{.+}}:           NULL

; CHECK:      0x0000[[BTOFF]]:  DW_TAG_base_type [4]  
; CHECK-NEXT:                     DW_AT_name [DW_FORM_strp] ({{.+}} = "int")

; CHECK:      0x{{.+}}:         NULL

; IR generated and reduced from:
; $ cat foo.cc
; struct Foo { int bar; };
; Foo foo;
; $ clang -g -S -emit-llvm foo.cc -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

%struct.Foo = type { i32 }

@foo = dso_local global %struct.Foo zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "foo.cc", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !3, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS3Foo")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "bar", scope: !6, file: !3, line: 1, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 12.0.0"}
