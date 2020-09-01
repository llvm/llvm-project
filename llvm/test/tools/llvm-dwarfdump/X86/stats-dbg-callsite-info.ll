; RUN: llc -emit-call-site-info -debug-entry-values %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s
;
; The LLVM IR file was generated on this source code by using
; option '-femit-debug-entry-values'.
;
; extern void foo(int *a, int b, int c, int d, int e, int f);
; extern int getVal();
;
; void baa(int arg1, int arg2, int arg3) {
;   int local1 = getVal();
;   foo(&local1, arg2, 10, 15, arg3 + 3, arg1 + arg2);
; }
;
; CHECK:      "#call site DIEs": 2,
; CHECK-NEXT: "#call site parameter DIEs": 6,
;
; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @baa(i32 %arg1, i32 %arg2, i32 %arg3) local_unnamed_addr #0 !dbg !10 {
entry:
  %local1 = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 %arg1, metadata !15, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 %arg2, metadata !16, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 %arg3, metadata !17, metadata !DIExpression()), !dbg !21
  %0 = bitcast i32* %local1 to i8*, !dbg !22
  %call = tail call i32 (...) @getVal(), !dbg !23
  call void @llvm.dbg.value(metadata i32 %call, metadata !18, metadata !DIExpression()), !dbg !24
  store i32 %call, i32* %local1, align 4, !dbg !24
  %add = add nsw i32 %arg3, 3, !dbg !24
  %add1 = add nsw i32 %arg2, %arg1, !dbg !24
  call void @llvm.dbg.value(metadata i32* %local1, metadata !18, metadata !DIExpression(DW_OP_deref)), !dbg !24
  call void @foo(i32* nonnull %local1, i32 %arg2, i32 10, i32 15, i32 %add, i32 %add1), !dbg !24
  ret void, !dbg !24
}

declare !dbg !4 dso_local i32 @getVal(...) local_unnamed_addr

declare !dbg !5 dso_local void @foo(i32*, i32, i32, i32, i32, i32) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/dir")
!2 = !{}
!3 = !{!4, !5}
!4 = !DISubprogram(name: "getVal", scope: !1, file: !1, line: 2, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang version 9.0.0"}
!10 = distinct !DISubprogram(name: "baa", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15, !16, !17, !18}
!15 = !DILocalVariable(name: "arg1", arg: 1, scope: !10, file: !1, line: 4, type: !13)
!16 = !DILocalVariable(name: "arg2", arg: 2, scope: !10, file: !1, line: 4, type: !13)
!17 = !DILocalVariable(name: "arg3", arg: 3, scope: !10, file: !1, line: 4, type: !13)
!18 = !DILocalVariable(name: "local1", scope: !10, file: !1, line: 5, type: !13)
!19 = !DILocation(line: 4, column: 14, scope: !10)
!20 = !DILocation(line: 4, column: 24, scope: !10)
!21 = !DILocation(line: 4, column: 34, scope: !10)
!22 = !DILocation(line: 5, column: 3, scope: !10)
!23 = !DILocation(line: 5, column: 16, scope: !10)
!24 = !DILocation(line: 5, column: 7, scope: !10)
