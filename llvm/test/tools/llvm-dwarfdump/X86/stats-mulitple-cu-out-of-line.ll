; RUN: llc -O0 %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s

; Test checks that statistics accounts two and more out-of-line instances
; of a function and reports the correct number of variables.

; $ cat test.h
;
; int foo(int a) { return a; }
;
; $ cat test1.cpp
;
; #include "test.h"
; int bar() { return foo(42); }
;
; $ cat test2.cpp
;
; #include "test.h"
; int far() { return foo(42); }

; CHECK:      "#functions": 3,
; CHECK-NEXT: "#functions with location": 3,
; CHECK-NEXT: "#inlined functions": 0,
; CHECK-NEXT: "#inlined functions with abstract origins": 0,
; CHECK-NEXT: "#unique source variables": 1,
; CHECK-NEXT: "#source variables": 2,
; CHECK-NEXT: "#source variables with location": 2,

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @_Z3fooi.1(i32 %a) !dbg !9 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !14, metadata !DIExpression()), !dbg !15
  %0 = load i32, ptr %a.addr, align 4, !dbg !15
  ret i32 %0, !dbg !15
}
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)
; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @_Z3barv() !dbg !18 {
entry:
  %call = call i32 @_Z3fooi.1(i32 42), !dbg !21
  ret i32 %call, !dbg !21
}
; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @_Z3fooi(i32 %a) !dbg !23 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !24, metadata !DIExpression()), !dbg !25
  %0 = load i32, ptr %a.addr, align 4, !dbg !25
  ret i32 %0, !dbg !25
}
; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @_Z3farv() !dbg !28 {
entry:
  %call = call i32 @_Z3fooi(i32 42), !dbg !29
  ret i32 %call, !dbg !29
}

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test1.cpp", directory: "/")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!4 = !DIFile(filename: "test2.cpp", directory: "/")
!5 = !{!"clang version 11.0.0"}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DIFile(filename: "./test.h", directory: "/")
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocalVariable(name: "a", arg: 1, scope: !9, file: !10, line: 1, type: !13)
!15 = !DILocation(line: 1, column: 13, scope: !9)
!18 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 3, type: !19, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!19 = !DISubroutineType(types: !20)
!20 = !{!13}
!21 = !DILocation(line: 3, column: 20, scope: !18)
!23 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !2)
!24 = !DILocalVariable(name: "a", arg: 1, scope: !23, file: !10, line: 1, type: !13)
!25 = !DILocation(line: 1, column: 13, scope: !23)
!28 = distinct !DISubprogram(name: "far", linkageName: "_Z3farv", scope: !4, file: !4, line: 3, type: !19, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !2)
!29 = !DILocation(line: 3, column: 20, scope: !28)
