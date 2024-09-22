; Test finding types by CompilerContext.
; RUN: llc %s -filetype=obj -o %t.o
; RUN: lldb-test symbols %t.o -find=type --language=C99 \
; RUN:   -compiler-context="Module:CModule,Module:SubModule,ClassOrStruct:FromSubmoduleX" \
; RUN:   | FileCheck %s --check-prefix=NORESULTS
; RUN: lldb-test symbols %t.o -find=type --language=C++ \
; RUN:   -compiler-context="Module:CModule,Module:SubModule,ClassOrStruct:FromSubmodule" \
; RUN:   | FileCheck %s --check-prefix=NORESULTS
; RUN: lldb-test symbols %t.o -find=type --language=C99 \
; RUN:   -compiler-context="Module:CModule,Module:SubModule,ClassOrStruct:FromSubmodule" \
; RUN:   | FileCheck %s
; RUN: lldb-test symbols %t.o -find=type -find-in-any-module --language=C99 \
; RUN:   -compiler-context="ClassOrStruct:FromSubmodule" \
; RUN:   | FileCheck %s
; RUN: lldb-test symbols %t.o -find=type --language=C99 \
; RUN:   -compiler-context="Module:CModule,Module:SubModule,AnyType:FromSubmodule" \
; RUN:   | FileCheck %s
;
; NORESULTS: Found 0 types
; CHECK: Found 1 types:
; CHECK: struct FromSubmodule {
; CHECK-NEXT:     unsigned int x;
; CHECK-NEXT:     unsigned int y;
; CHECK-NEXT:     unsigned int z;
; CHECK-NEXT: }

source_filename = "/t.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

!llvm.dbg.cu = !{!2}
!llvm.linker.options = !{}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!22}

; This simulates the debug info for a Clang module.
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: GNU, retainedTypes: !{!11}, sysroot: "/")
!3 = !DIFile(filename: "t.c", directory: "/")
!8 = !DIModule(scope: !9, name: "SubModule", includePath: "")
!9 = !DIModule(scope: null, name: "CModule", includePath: "")
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "FromSubmodule", scope: !8, file: !3, line: 1, size: 96, elements: !13)
!13 = !{!14, !16, !17}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !11, file: !3, line: 2, baseType: !15, size: 32)
!15 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !11, file: !3, line: 2, baseType: !15, size: 32, offset: 32)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !11, file: !3, line: 2, baseType: !15, size: 32, offset: 64)
!18 = !{i32 2, !"Dwarf Version", i32 4}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project 056f1b5cc7c2133f0cb3e30e7f24808d321096d7)"}
