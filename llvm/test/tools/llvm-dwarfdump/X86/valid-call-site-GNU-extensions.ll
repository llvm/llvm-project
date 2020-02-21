; RUN: llc -debug-entry-values %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -verify - | FileCheck %s
;
; CHECK: No errors.
;
; The source code of the test case:
;
;extern int fn1();
;extern int fn3(int);
;
;__attribute__((noinline))
;void
;fn2 (int *arg) {
;  int a = ++(*arg);
;  fn3 (a);
;}
;
;__attribute__((noinline))
;int f() {
;  int x = fn1();
;  fn2 (&x);
;  return 0;
;}
;
; ModuleID = 'test.c'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local void @fn2(i32* nocapture %arg) local_unnamed_addr !dbg !15 {
entry:
  call void @llvm.dbg.value(metadata i32* %arg, metadata !20, metadata !DIExpression()), !dbg !22
  %0 = load i32, i32* %arg, align 4, !dbg !23
  %inc = add nsw i32 %0, 1, !dbg !23
  store i32 %inc, i32* %arg, align 4, !dbg !23
  call void @llvm.dbg.value(metadata i32 %inc, metadata !21, metadata !DIExpression()), !dbg !22
  %call = tail call i32 @fn3(i32 %inc), !dbg !23
  ret void, !dbg !23
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare !dbg !4 dso_local i32 @fn3(i32) local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @f() local_unnamed_addr !dbg !24 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*, !dbg !29
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0), !dbg !29
  %call = tail call i32 (...) @fn1() #4, !dbg !29
  call void @llvm.dbg.value(metadata i32 %call, metadata !28, metadata !DIExpression()), !dbg !30
  store i32 %call, i32* %x, align 4, !dbg !29
  call void @llvm.dbg.value(metadata i32* %x, metadata !28, metadata !DIExpression(DW_OP_deref)), !dbg !30
  call void @fn2(i32* nonnull %x), !dbg !29
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0), !dbg !29
  ret i32 0, !dbg !29
}

declare !dbg !8 dso_local i32 @fn1(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "dir")
!2 = !{}
!3 = !{!4, !8}
!4 = !DISubprogram(name: "fn3", scope: !1, file: !1, line: 2, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !9, spFlags: DISPFlagOptimized, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!7, null}
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 10.0.0"}
!15 = distinct !DISubprogram(name: "fn2", scope: !1, file: !1, line: 6, type: !16, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!19 = !{!20, !21}
!20 = !DILocalVariable(name: "arg", arg: 1, scope: !15, file: !1, line: 6, type: !18)
!21 = !DILocalVariable(name: "a", scope: !15, file: !1, line: 7, type: !7)
!22 = !DILocation(line: 0, scope: !15)
!23 = !DILocation(line: 7, column: 11, scope: !15)
!24 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 12, type: !25, scopeLine: 12, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !27)
!25 = !DISubroutineType(types: !17)
!26 = !{!7}
!27 = !{!28}
!28 = !DILocalVariable(name: "x", scope: !24, file: !1, line: 13, type: !7)
!29 = !DILocation(line: 13, column: 3, scope: !24)
!30 = !DILocation(line: 0, scope: !24)
