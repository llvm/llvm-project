; RUN: llc -filetype=obj %s -o %t.obj
; RUN: llvm-pdbutil dump -symbols -types %t.obj | FileCheck %s

; Make sure that `gv`, `Bar`, and `Baz` all point to the complete type index of
; Foo (0x1002), and not the forward declaration index (0x1000).

; C++ source:
; struct Foo {
;   int x;
; };
; typedef Foo Bar;
; typedef Bar Baz;
; static_assert(sizeof(Foo) == 4, "");
; Baz gv;

; CHECK:                       Types (.debug$T)
; CHECK-NEXT: ============================================================
; CHECK: 0x1000 | LF_STRUCTURE [size = 36] `Foo`
; CHECK:          unique name: `.?AUFoo@@`
; CHECK:          vtable: <no type>, base list: <no type>, field list: <no type>
; CHECK:          options: forward ref | has unique name, sizeof 0
; CHECK: 0x1001 | LF_FIELDLIST [size = 16]
; CHECK:          - LF_MEMBER [name = `x`, Type = 0x0074 (int), offset = 0, attrs = public]
; CHECK: 0x1002 | LF_STRUCTURE [size = 36] `Foo`
; CHECK:          unique name: `.?AUFoo@@`
; CHECK:          vtable: <no type>, base list: <no type>, field list: 0x1001
; CHECK:          options: has unique name, sizeof 4
; CHECK: 0x1004 | LF_UDT_SRC_LINE [size = 16]
; CHECK:          udt = 0x1002, file = 4099, line = 1

; CHECK:                           Symbols
; CHECK: ============================================================
; CHECK:   Mod 0000 | `.debug$S`:
; CHECK:        0 | S_GDATA32 [size = 20] `gv`
; CHECK:            type = 0x1002 (Foo), addr = 0000:0000
; CHECK:        0 | S_UDT [size = 12] `Bar`
; CHECK:            original type = 0x1002
; CHECK:        0 | S_UDT [size = 12] `Baz`
; CHECK:            original type = 0x1002
; CHECK:        0 | S_UDT [size = 12] `Foo`
; CHECK:            original type = 0x1002

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.14.26433"

%struct.Foo = type { i32 }

@"?gv@@3UFoo@@A" = dso_local global %struct.Foo zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "gv", linkageName: "?gv@@3UFoo@@A", scope: !2, file: !3, line: 7, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 8.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "e21cd263d43db2ff55050be3e41e789f")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "Baz", file: !3, line: 5, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar", file: !3, line: 4, baseType: !8)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !3, line: 1, size: 32, flags: DIFlagTypePassByValue | DIFlagTrivial, elements: !9, identifier: ".?AUFoo@@")
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !8, file: !3, line: 2, baseType: !11, size: 32)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 2, !"CodeView", i32 1}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 2}
!15 = !{i32 7, !"PIC Level", i32 2}
!16 = !{!"clang version 8.0.0 "}
