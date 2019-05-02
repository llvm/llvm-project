; Check that debug info for nested types is generated correctly.
; With full LTO we may end up in several compilation units sharing
; single object file. In such case it's possible that nested type DIE
; is created on behalf of CU#1 and parent type DIE is created on 
; behalf of CU#2. In such case we may end up with broken source line
; attributes, because file ids don't coincide in CU#1 and CU#2

; Sources used to generate this sample (C++)
; // File: inc1.h
; struct S {
;   struct Nested {};
;   typedef int NestedTypedef;
; };
;
; // File inc2.h
; extern int x;
; int x;
; 
; // File foo.cpp
; #include "inc1.h"
; S s;
;
; // File bar.cpp
; #include "inc1.h"
; #include "inc2.h"
; void f3() {
;   S::Nested n;
;   S::NestedTypedef n2;
; }

; RUN: llc -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK:             DW_AT_name ("Nested")
; CHECK-NEXT:        DW_AT_byte_size (0x01)
; CHECK-NEXT:        DW_AT_decl_file ("{{.*}}inc1.h")
; CHECK-NEXT:        DW_AT_decl_line (2)

; CHECK:             DW_AT_name ("NestedTypedef")
; CHECK-NEXT:        DW_AT_decl_file ("{{.*}}inc1.h")
; CHECK-NEXT:        DW_AT_decl_line (3)

source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i8 }

@s = dso_local global %struct.S zeroinitializer, align 1, !dbg !0
@x = dso_local global i32 0, align 4, !dbg !8

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z2f3v() !dbg !19 {
entry:
  %n = alloca %struct.S, align 1
  %n2 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata %struct.S* %n, metadata !22, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata i32* %n2, metadata !25, metadata !DIExpression()), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2, !10}
!llvm.ident = !{!15, !15}
!llvm.module.flags = !{!16, !17, !18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (trunk 354767)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "foo.cpp", directory: "/home/evgeny/work/cpp_lexer/sample3")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !7, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !4, identifier: "_ZTS1S")
!7 = !DIFile(filename: "./inc1.h", directory: "/home/evgeny/work/cpp_lexer/sample3")
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "x", scope: !10, file: !13, line: 2, type: !14, isLocal: false, isDefinition: true)
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !11, producer: "clang version 9.0.0 (trunk 354767)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !12, nameTableKind: None)
!11 = !DIFile(filename: "bar.cpp", directory: "/home/evgeny/work/cpp_lexer/sample3")
!12 = !{!8}
!13 = !DIFile(filename: "./inc2.h", directory: "/home/evgeny/work/cpp_lexer/sample3")
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!"clang version 9.0.0 (trunk 354767)"}
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !11, file: !11, line: 3, type: !20, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !4)
!20 = !DISubroutineType(types: !21)
!21 = !{null}
!22 = !DILocalVariable(name: "n", scope: !19, file: !11, line: 4, type: !23)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Nested", scope: !6, file: !7, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !4, identifier: "_ZTSN1S6NestedE")
!24 = !DILocation(line: 4, column: 13, scope: !19)
!25 = !DILocalVariable(name: "n2", scope: !19, file: !11, line: 5, type: !26)
!26 = !DIDerivedType(tag: DW_TAG_typedef, name: "NestedTypedef", scope: !6, file: !7, line: 3, baseType: !14)
!27 = !DILocation(line: 5, column: 20, scope: !19)
!28 = !DILocation(line: 6, column: 1, scope: !19)

