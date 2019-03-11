; Checks that DW_AT_decl_file attribute is correctly set for
; template function declaration. When using full LTO we likely
; end up having multiple CUs in a single DWARF file, and it is
; possible that we add template function declaration DIE from
; CU#1 to structure type DIE from CU#2. In such case we should
; add source line attributes on behalf of CU#2, not CU#1

; Example source code used to generate this sample (C++)
; // File inc1.h
; struct S {
;   template <typename T> static void tmpfn() {}
; };
;
; // File: inc2.h
; extern int x;
; int x;
;
; // File: foo.cpp
; #include "inc1.h"
; S s;
;
; // File: bar.cpp
; #include "inc1.h"
; #include "inc2.h"
; void f3() { S::tmpfn<int>(); }

; RUN: llc -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t -o - | FileCheck %s

; CHECK:       DW_AT_linkage_name ("_ZN1S5tmpfnIiEEvv")
; CHECK-NEXT:  DW_AT_name ("tmpfn<int>")
; CHECK-NEXT:  DW_AT_decl_file ("{{.*}}inc1.h")
; CHECK-NEXT:  DW_AT_decl_line (2)

source_filename = "ld-temp.o"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i8 }

$_ZN1S5tmpfnIiEEvv = comdat any

@s = dso_local global %struct.S zeroinitializer, align 1, !dbg !0
@x = dso_local global i32 0, align 4, !dbg !8

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z2f3v() !dbg !21 {
  call void @_ZN1S5tmpfnIiEEvv(), !dbg !24
  ret void, !dbg !25
}

; Function Attrs: noinline nounwind optnone uwtable
define weak_odr dso_local void @_ZN1S5tmpfnIiEEvv() comdat align 2 !dbg !26 {
  ret void, !dbg !30
}

!llvm.dbg.cu = !{!2, !10}
!llvm.ident = !{!15, !15}
!llvm.module.flags = !{!16, !17, !18, !19, !20}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (trunk 354767)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "foo.cpp", directory: "/home/evgeny/work/cpp_lexer/sample2")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !7, line: 1, size: 8, flags: DIFlagTypePassByValue | DIFlagTrivial, elements: !4, identifier: "_ZTS1S")
!7 = !DIFile(filename: "./inc1.h", directory: "/home/evgeny/work/cpp_lexer/sample2")
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "x", scope: !10, file: !13, line: 2, type: !14, isLocal: false, isDefinition: true)
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !11, producer: "clang version 9.0.0 (trunk 354767)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !12, nameTableKind: None)
!11 = !DIFile(filename: "bar.cpp", directory: "/home/evgeny/work/cpp_lexer/sample2")
!12 = !{!8}
!13 = !DIFile(filename: "./inc2.h", directory: "/home/evgeny/work/cpp_lexer/sample2")
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!"clang version 9.0.0 (trunk 354767)"}
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{i32 1, !"ThinLTO", i32 0}
!20 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!21 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !11, file: !11, line: 3, type: !22, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !4)
!22 = !DISubroutineType(types: !23)
!23 = !{null}
!24 = !DILocation(line: 3, column: 13, scope: !21)
!25 = !DILocation(line: 3, column: 30, scope: !21)
!26 = distinct !DISubprogram(name: "tmpfn<int>", linkageName: "_ZN1S5tmpfnIiEEvv", scope: !6, file: !7, line: 2, type: !22, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, templateParams: !28, declaration: !27, retainedNodes: !4)
!27 = !DISubprogram(name: "tmpfn<int>", linkageName: "_ZN1S5tmpfnIiEEvv", scope: !6, file: !7, line: 2, type: !22, scopeLine: 2, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0, templateParams: !28)
!28 = !{!29}
!29 = !DITemplateTypeParameter(name: "T", type: !14)
!30 = !DILocation(line: 2, column: 46, scope: !26)

