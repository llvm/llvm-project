; RUN: llc -debug-entry-values %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -statistics - | FileCheck %s
;
; CHECK: "entry value scope bytes covered":5
; CHECK: "formal params scope bytes total":20
; CHECK: "formal params scope bytes covered":20
; CHECK: "formal params entry value scope bytes covered":5
; CHECK: "vars scope bytes total":78
; CHECK: "vars scope bytes covered":60
; CHECK: "vars entry value scope bytes covered":0
; CHECK: "total variables procesed by location statistics":6
; CHECK: "variables with 0% of its scope covered":1
; CHECK: "variables with 1-9% of its scope covered":0
; CHECK: "variables with 10-19% of its scope covered":0
; CHECK: "variables with 20-29% of its scope covered":0
; CHECK: "variables with 30-39% of its scope covered":0
; CHECK: "variables with 40-49% of its scope covered":0
; CHECK: "variables with 50-59% of its scope covered":1
; CHECK: "variables with 60-69% of its scope covered":0
; CHECK: "variables with 70-79% of its scope covered":0
; CHECK: "variables with 80-89% of its scope covered":1
; CHECK: "variables with 90-99% of its scope covered":0
; CHECK: "variables with 100% of its scope covered":3
; CHECK: "variables (excluding the debug entry values) with 0% of its scope covered":1
; CHECK: "variables (excluding the debug entry values) with 1-9% of its scope covered":0
; CHECK: "variables (excluding the debug entry values) with 10-19% of its scope covered":0
; CHECK: "variables (excluding the debug entry values) with 20-29% of its scope covered":0
; CHECK: "variables (excluding the debug entry values) with 30-39% of its scope covered":0
; CHECK: "variables (excluding the debug entry values) with 40-49% of its scope covered":0
; CHECK: "variables (excluding the debug entry values) with 50-59% of its scope covered":2
; CHECK: "variables (excluding the debug entry values) with 60-69% of its scope covered":0
; CHECK: "variables (excluding the debug entry values) with 70-79% of its scope covered":0
; CHECK: "variables (excluding the debug entry values) with 80-89% of its scope covered":1
; CHECK: "variables (excluding the debug entry values) with 90-99% of its scope covered":0
; CHECK: "variables (excluding the debug entry values) with 100% of its scope covered":2
; CHECK: "total params procesed by location statistics":2
; CHECK: "params with 0% of its scope covered":0
; CHECK: "params with 1-9% of its scope covered":0
; CHECK: "params with 10-19% of its scope covered":0
; CHECK: "params with 20-29% of its scope covered":0
; CHECK: "params with 30-39% of its scope covered":0
; CHECK: "params with 40-49% of its scope covered":0
; CHECK: "params with 50-59% of its scope covered":0
; CHECK: "params with 60-69% of its scope covered":0
; CHECK: "params with 70-79% of its scope covered":0
; CHECK: "params with 80-89% of its scope covered":0
; CHECK: "params with 90-99% of its scope covered":0
; CHECK: "params with 100% of its scope covered":2
; CHECK: "params (excluding the debug entry values) with 0% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 1-9% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 10-19% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 20-29% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 30-39% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 40-49% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 50-59% of its scope covered":1
; CHECK: "params (excluding the debug entry values) with 60-69% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 70-79% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 80-89% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 90-99% of its scope covered":0
; CHECK: "params (excluding the debug entry values) with 100% of its scope covered":1
; CHECK: "total vars procesed by location statistics":4
; CHECK: "vars with 0% of its scope covered":1
; CHECK: "vars with 1-9% of its scope covered":0
; CHECK: "vars with 10-19% of its scope covered":0
; CHECK: "vars with 20-29% of its scope covered":0
; CHECK: "vars with 30-39% of its scope covered":0
; CHECK: "vars with 40-49% of its scope covered":0
; CHECK: "vars with 50-59% of its scope covered":1
; CHECK: "vars with 60-69% of its scope covered":0
; CHECK: "vars with 70-79% of its scope covered":0
; CHECK: "vars with 80-89% of its scope covered":1
; CHECK: "vars with 90-99% of its scope covered":0
; CHECK: "vars with 100% of its scope covered":1
; CHECK: "vars (excluding the debug entry values) with 0% of its scope covered":1
; CHECK: "vars (excluding the debug entry values) with 1-9% of its scope covered":0
; CHECK: "vars (excluding the debug entry values) with 10-19% of its scope covered":0
; CHECK: "vars (excluding the debug entry values) with 20-29% of its scope covered":0
; CHECK: "vars (excluding the debug entry values) with 30-39% of its scope covered":0
; CHECK: "vars (excluding the debug entry values) with 40-49% of its scope covered":0
; CHECK: "vars (excluding the debug entry values) with 50-59% of its scope covered":1
; CHECK: "vars (excluding the debug entry values) with 60-69% of its scope covered":0
; CHECK: "vars (excluding the debug entry values) with 70-79% of its scope covered":0
; CHECK: "vars (excluding the debug entry values) with 80-89% of its scope covered":1
; CHECK: "vars (excluding the debug entry values) with 90-99% of its scope covered":0
; CHECK: "vars (excluding the debug entry values) with 100% of its scope covered":1
;
; The source code of the test case:
; extern void fn3(int *);
; extern void fn2 (int);
; __attribute__((noinline))
; void
; fn1 (int x, int y)
; {
;   int u = x + y;
;   if (x > 1)
;     u += 1;
;   else
;     u += 2;
;   if (y > 4)
;     u += x;
;   int a = 7;
;   fn2 (a);
;   u --;
; }
;
; __attribute__((noinline))
; int f()
; {
;   int l, k;
;   fn3(&l);
;   fn3(&k);
;   fn1 (l, k);
;   return 0;
; }
;
; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local void @fn1(i32 %x, i32 %y) local_unnamed_addr !dbg !16 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !20, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 %y, metadata !21, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 undef, metadata !22, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 undef, metadata !22, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 7, metadata !23, metadata !DIExpression()), !dbg !24
  tail call void @fn2(i32 7), !dbg !25
  call void @llvm.dbg.value(metadata i32 undef, metadata !22, metadata !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !24
  ret void, !dbg !26
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare !dbg !4 dso_local void @fn2(i32) local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @f() local_unnamed_addr !dbg !27 {
entry:
  %l = alloca i32, align 4
  %k = alloca i32, align 4
  %0 = bitcast i32* %l to i8*, !dbg !33
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0), !dbg !33
  %1 = bitcast i32* %k to i8*, !dbg !33
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %1), !dbg !33
  call void @llvm.dbg.value(metadata i32* %l, metadata !31, metadata !DIExpression(DW_OP_deref)), !dbg !34
  call void @fn3(i32* nonnull %l), !dbg !35
  call void @llvm.dbg.value(metadata i32* %k, metadata !32, metadata !DIExpression(DW_OP_deref)), !dbg !34
  call void @fn3(i32* nonnull %k), !dbg !36
  %2 = load i32, i32* %l, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %2, metadata !31, metadata !DIExpression()), !dbg !34
  %3 = load i32, i32* %k, align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %3, metadata !32, metadata !DIExpression()), !dbg !34
  call void @fn1(i32 %2, i32 %3), !dbg !37
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %1), !dbg !37
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0), !dbg !37
  ret i32 0, !dbg !37
}

declare !dbg !8 dso_local void @fn3(i32*) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{!4, !8}
!4 = !DISubprogram(name: "fn2", scope: !1, file: !1, line: 2, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DISubprogram(name: "fn3", scope: !1, file: !1, line: 1, type: !9, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 10.0.0"}
!16 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 6, type: !17, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !7, !7}
!19 = !{!20, !21, !22, !23}
!20 = !DILocalVariable(name: "x", arg: 1, scope: !16, file: !1, line: 6, type: !7)
!21 = !DILocalVariable(name: "y", arg: 2, scope: !16, file: !1, line: 6, type: !7)
!22 = !DILocalVariable(name: "u", scope: !16, file: !1, line: 8, type: !7)
!23 = !DILocalVariable(name: "a", scope: !16, file: !1, line: 18, type: !7)
!24 = !DILocation(line: 0, scope: !16)
!25 = !DILocation(line: 20, column: 3, scope: !16)
!26 = !DILocation(line: 22, column: 1, scope: !16)
!27 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 25, type: !28, scopeLine: 26, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !30)
!28 = !DISubroutineType(types: !29)
!29 = !{!7}
!30 = !{!31, !32}
!31 = !DILocalVariable(name: "l", scope: !27, file: !1, line: 27, type: !7)
!32 = !DILocalVariable(name: "k", scope: !27, file: !1, line: 27, type: !7)
!33 = !DILocation(line: 27, column: 3, scope: !27)
!34 = !DILocation(line: 0, scope: !27)
!35 = !DILocation(line: 29, column: 3, scope: !27)
!36 = !DILocation(line: 30, column: 3, scope: !27)
!37 = !DILocation(line: 32, column: 8, scope: !27)
