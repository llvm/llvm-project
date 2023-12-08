; RUN: llc -O0 %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s

; This checks that if DW_TAG_structure_type duplicates in multiple CU,
; the number of total vars ("source variables") will include every copy,
; so the number of "variables with location" doesn't exceed the number of total vars.

; $ cat test.h
; struct s { static const int ss = 42; };
;
; $ cat test1.cpp
; #include "test.h"
; s S1;
;
; $ cat test2.cpp
; #include "test.h"
; s S2;

; CHECK:      "#source variables": 4,
; CHECK:      "#source variables with location": 4,

source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.s = type { i8 }
%struct.s.1 = type { i8 }

@S1 = dso_local global %struct.s zeroinitializer, align 1, !dbg !0
@S2 = dso_local global %struct.s.1 zeroinitializer, align 1, !dbg !13

!llvm.dbg.cu = !{!2, !15}
!llvm.ident = !{!18, !18}
!llvm.module.flags = !{!19, !20, !21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "S1", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !12, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test1.cpp", directory: "/")
!4 = !{}
!5 = !{!6}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !7, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !8, identifier: "_ZTS1s")
!7 = !DIFile(filename: "./test.h", directory: "/")
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "ss", scope: !6, file: !7, line: 1, baseType: !10, flags: DIFlagStaticMember, extraData: i32 42)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!0}
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "S2", scope: !15, file: !16, line: 2, type: !6, isLocal: false, isDefinition: true)
!15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !16, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !22, globals: !17, splitDebugInlining: false, nameTableKind: None)
!16 = !DIFile(filename: "test2.cpp", directory: "/")
!17 = !{!13}
!18 = !{!"clang version 10.0.0"}
!19 = !{i32 7, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"wchar_size", i32 4}
!22 = !{!23}
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !7, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !24, identifier: "_ZTS1s")
!24 = !{!25}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "ss", scope: !23, file: !7, line: 1, baseType: !10, flags: DIFlagStaticMember, extraData: i32 42)
