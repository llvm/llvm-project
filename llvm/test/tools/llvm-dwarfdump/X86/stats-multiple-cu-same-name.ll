; RUN: llc -O0 %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s

; Test that statistics distinguish functions with the same name.

; CHECK:      "#functions": 4,
; CHECK:      "#unique source variables": 2,
; CHECK-NEXT: "#source variables": 2,

; $ cat test1.cpp
; static int foo(int a) {
;   return a;
; }
; int boo() { return foo(42); }
;
; $ cat test2.cpp
; static int foo(int a) {
;   return a;
; }
; int bar() { return foo(42); }

source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline optnone uwtable
define dso_local i32 @_Z3boov() !dbg !9 {
entry:
  %call = call i32 @_ZL3fooi(i32 42), !dbg !13
  ret i32 %call
}
; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @_ZL3fooi(i32 %a) !dbg !15 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !18, metadata !DIExpression()), !dbg !19
  %0 = load i32, i32* %a.addr, align 4, !dbg !20
  ret i32 %0
}
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)
; Function Attrs: noinline optnone uwtable
define dso_local i32 @_Z3barv() !dbg !22 {
entry:
  %call = call i32 @_ZL3fooi.1(i32 442), !dbg !23
  ret i32 %call
}
; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @_ZL3fooi.1(i32 %a) !dbg !25 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !26, metadata !DIExpression()), !dbg !27
  %0 = load i32, i32* %a.addr, align 4, !dbg !28
  %mul = mul nsw i32 %0, 2
  ret i32 %mul
}

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test1.cpp", directory: "/")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!4 = !DIFile(filename: "test2.cpp", directory: "/")
!5 = !{!"clang version 10.0.0"}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = distinct !DISubprogram(name: "boo", linkageName: "_Z3boov", scope: !1, file: !1, line: 5, type: !10, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 6, column: 10, scope: !9)
!15 = distinct !DISubprogram(name: "foo", linkageName: "_ZL3fooi", scope: !1, file: !1, line: 1, type: !16, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{!12, !12}
!18 = !DILocalVariable(name: "a", arg: 1, scope: !15, file: !1, line: 1, type: !12)
!19 = !DILocation(line: 1, column: 20, scope: !15)
!20 = !DILocation(line: 2, column: 10, scope: !15)
!22 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !4, file: !4, line: 5, type: !10, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !2)
!23 = !DILocation(line: 6, column: 10, scope: !22)
!25 = distinct !DISubprogram(name: "foo", linkageName: "_ZL3fooi", scope: !4, file: !4, line: 1, type: !16, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !3, retainedNodes: !2)
!26 = !DILocalVariable(name: "a", arg: 1, scope: !25, file: !4, line: 1, type: !12)
!27 = !DILocation(line: 1, column: 20, scope: !25)
!28 = !DILocation(line: 2, column: 10, scope: !25)
