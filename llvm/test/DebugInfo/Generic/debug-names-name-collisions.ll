; RUN: %llc_dwarf -accel-tables=Dwarf -dwarf-linkage-names=All -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s

; Generated from the following C code using
; clang -S -emit-llvm col.cc
;
; namespace foo { struct foo {}; struct foo foo; }
; namespace bar { struct bar {}; struct bar bar; }
; namespace baz { struct baz {}; struct baz baz; }

; We have 6 names: foo, bar, baz and three mangled names of the variables.
; CHECK: Name count: 6

; Check that all the names are present in the output correct number of times.
; CHECK: String: 0x{{[0-9a-f]*}} "bar"
; CHECK-DAG: Tag: DW_TAG_namespace
; CHECK-DAG: Tag: DW_TAG_variable
; CHECK-DAG: Tag: DW_TAG_structure_type
; CHECK: String: 0x{{[0-9a-f]*}} "baz"
; CHECK-DAG: Tag: DW_TAG_namespace
; CHECK-DAG: Tag: DW_TAG_variable
; CHECK-DAG: Tag: DW_TAG_structure_type
; CHECK: String: 0x{{[0-9a-f]*}} "foo"
; CHECK-DAG: Tag: DW_TAG_namespace
; CHECK-DAG: Tag: DW_TAG_variable
; CHECK-DAG: Tag: DW_TAG_structure_type
; CHECK: String: 0x{{[0-9a-f]*}} "_ZN3foo3fooE"
; CHECK:   Tag: DW_TAG_variable
; CHECK: String: 0x{{[0-9a-f]*}} "_ZN3bar3barE"
; CHECK:   Tag: DW_TAG_variable
; CHECK: String: 0x{{[0-9a-f]*}} "_ZN3baz3bazE"
; CHECK:   Tag: DW_TAG_variable

; VERIFY: No errors.

%"struct.foo::foo" = type { i8 }
%"struct.bar::bar" = type { i8 }
%"struct.baz::baz" = type { i8 }

@_ZN3foo3fooE = dso_local global %"struct.foo::foo" zeroinitializer, align 1, !dbg !0
@_ZN3bar3barE = dso_local global %"struct.bar::bar" zeroinitializer, align 1, !dbg !6
@_ZN3baz3bazE = dso_local global %"struct.baz::baz" zeroinitializer, align 1, !dbg !10

!llvm.dbg.cu = !{!14}
!llvm.module.flags = !{!16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", linkageName: "_ZN3foo3fooE", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = !DINamespace(name: "foo", scope: null)
!3 = !DIFile(filename: "/tmp/col.cc", directory: "/tmp")
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", scope: !2, file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !5, identifier: "_ZTSN3foo3fooE")
!5 = !{}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "bar", linkageName: "_ZN3bar3barE", scope: !8, file: !3, line: 2, type: !9, isLocal: false, isDefinition: true)
!8 = !DINamespace(name: "bar", scope: null)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", scope: !8, file: !3, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !5, identifier: "_ZTSN3bar3barE")
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "baz", linkageName: "_ZN3baz3bazE", scope: !12, file: !3, line: 3, type: !13, isLocal: false, isDefinition: true)
!12 = !DINamespace(name: "baz", scope: null)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "baz", scope: !12, file: !3, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !5, identifier: "_ZTSN3baz3bazE")
!14 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, globals: !15)
!15 = !{!0, !6, !10}
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{!"clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)"}
