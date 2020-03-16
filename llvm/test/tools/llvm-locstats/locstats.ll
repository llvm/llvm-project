; UNSUPPORTED: system-windows
; REQUIRES: x86-registered-target
; RUN: llc %s -o %t0.o -filetype=obj
; RUN: %llvm-locstats %t0.o | FileCheck %s --check-prefix=LOCSTATS
;
; Test the llvm-locstats output.
; LOCSTATS: 0% 0 0%
; LOCSTATS: (0%,10%) 0 0%
; LOCSTATS: [10%,20%) 0 0%
; LOCSTATS: [20%,30%) 1 11%
; LOCSTATS: [30%,40%) 0 0%
; LOCSTATS: [40%,50%) 1 11%
; LOCSTATS: [50%,60%) 1 11%
; LOCSTATS: [60%,70%) 1 11%
; LOCSTATS: [70%,80%) 0 0%
; LOCSTATS: [80%,90%) 2 22%
; LOCSTATS: [90%,100%) 1 11%
; LOCSTATS: 100% 2 22%
;
; The source code of the test case:
;extern int fn2 (int);
;
;__attribute__((noinline))
;int
;fn1 (int *x, int *y)
;{
;  int a = *x;
;  int b = *y;
;  int local = a + b;
;  if (a > 1) {
;    local += 2;
;    ++local;
;    if (local > 200)
;      local -= fn2(a);
;  } else {
;    local += 3;
;    ++local;
;    local += fn2(a);
; }
;  if (b > 4)
;   local += a;
;  int local2 = 7;
;  local -= fn2 (local2);
;  return local;
;}
;
;__attribute__((noinline))
;int f()
;{
;  int l, k;
;  int res = 0;
;  res += fn1 (&l, &k);
;  return res;
;}
;
; ModuleID = 'locstats.c'
source_filename = "locstats.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @fn1(i32* nocapture readonly %0, i32* nocapture readonly %1) local_unnamed_addr !dbg !7 {
  call void @llvm.dbg.value(metadata i32* %0, metadata !13, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32* %1, metadata !14, metadata !DIExpression()), !dbg !19
  %3 = load i32, i32* %0, align 4, !dbg !20
  call void @llvm.dbg.value(metadata i32 %3, metadata !15, metadata !DIExpression()), !dbg !19
  %4 = load i32, i32* %1, align 4, !dbg !20
  call void @llvm.dbg.value(metadata i32 %4, metadata !16, metadata !DIExpression()), !dbg !19
  %5 = add nsw i32 %4, %3, !dbg !20
  call void @llvm.dbg.value(metadata i32 %5, metadata !17, metadata !DIExpression()), !dbg !19
  %6 = icmp sgt i32 %3, 1, !dbg !20
  br i1 %6, label %7, label %13, !dbg !22

7:                                                ; preds = %2
  call void @llvm.dbg.value(metadata i32 %5, metadata !17, metadata !DIExpression(DW_OP_plus_uconst, 2, DW_OP_stack_value)), !dbg !19
  %8 = add nsw i32 %5, 3, !dbg !23
  call void @llvm.dbg.value(metadata i32 %8, metadata !17, metadata !DIExpression()), !dbg !19
  %9 = icmp sgt i32 %8, 200, !dbg !25
  br i1 %9, label %10, label %17, !dbg !27

10:                                               ; preds = %7
  %11 = tail call i32 @fn2(i32 %3), !dbg !27
  %12 = sub nsw i32 %8, %11, !dbg !27
  call void @llvm.dbg.value(metadata i32 %12, metadata !17, metadata !DIExpression()), !dbg !19
  br label %17, !dbg !27

13:                                               ; preds = %2
  call void @llvm.dbg.value(metadata i32 %5, metadata !17, metadata !DIExpression(DW_OP_plus_uconst, 3, DW_OP_stack_value)), !dbg !19
  %14 = add nsw i32 %5, 4, !dbg !28
  call void @llvm.dbg.value(metadata i32 %14, metadata !17, metadata !DIExpression()), !dbg !19
  %15 = tail call i32 @fn2(i32 %3), !dbg !30
  %16 = add nsw i32 %14, %15, !dbg !30
  call void @llvm.dbg.value(metadata i32 %16, metadata !17, metadata !DIExpression()), !dbg !19
  br label %17

17:                                               ; preds = %7, %10, %13
  %18 = phi i32 [ %12, %10 ], [ %8, %7 ], [ %16, %13 ], !dbg !31
  call void @llvm.dbg.value(metadata i32 %18, metadata !17, metadata !DIExpression()), !dbg !19
  %19 = icmp sgt i32 %4, 4, !dbg !32
  %20 = select i1 %19, i32 %3, i32 0, !dbg !34
  %21 = add nsw i32 %18, %20, !dbg !34
  call void @llvm.dbg.value(metadata i32 %21, metadata !17, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 7, metadata !18, metadata !DIExpression()), !dbg !19
  %22 = tail call i32 @fn2(i32 7), !dbg !34
  %23 = sub i32 %21, %22, !dbg !34
  call void @llvm.dbg.value(metadata i32 %23, metadata !17, metadata !DIExpression()), !dbg !19
  ret i32 %23, !dbg !34
}

declare dso_local i32 @fn2(i32) local_unnamed_addr

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @f() local_unnamed_addr !dbg !35 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = bitcast i32* %1 to i8*, !dbg !42
  %4 = bitcast i32* %2 to i8*, !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !41, metadata !DIExpression()), !dbg !42
  call void @llvm.dbg.value(metadata i32* %1, metadata !39, metadata !DIExpression(DW_OP_deref)), !dbg !42
  call void @llvm.dbg.value(metadata i32* %2, metadata !40, metadata !DIExpression(DW_OP_deref)), !dbg !42
  %5 = call i32 @fn1(i32* nonnull %1, i32* nonnull %2), !dbg !42
  call void @llvm.dbg.value(metadata i32 %5, metadata !41, metadata !DIExpression()), !dbg !42
  ret i32 %5, !dbg !42
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "locstats.c", directory: "/dir")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0"}
!7 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 5, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!12 = !{!13, !14, !15, !16, !17, !18}
!13 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !1, line: 5, type: !11)
!14 = !DILocalVariable(name: "y", arg: 2, scope: !7, file: !1, line: 5, type: !11)
!15 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 7, type: !10)
!16 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 8, type: !10)
!17 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 9, type: !10)
!18 = !DILocalVariable(name: "local2", scope: !7, file: !1, line: 22, type: !10)
!19 = !DILocation(line: 0, scope: !7)
!20 = !DILocation(line: 7, column: 11, scope: !7)
!21 = distinct !DILexicalBlock(scope: !7, file: !1, line: 10, column: 7)
!22 = !DILocation(line: 10, column: 7, scope: !7)
!23 = !DILocation(line: 12, column: 5, scope: !24)
!24 = distinct !DILexicalBlock(scope: !21, file: !1, line: 10, column: 14)
!25 = !DILocation(line: 13, column: 15, scope: !26)
!26 = distinct !DILexicalBlock(scope: !24, file: !1, line: 13, column: 9)
!27 = !DILocation(line: 13, column: 9, scope: !24)
!28 = !DILocation(line: 17, column: 5, scope: !26)
!29 = distinct !DILexicalBlock(scope: !21, file: !1, line: 15, column: 10)
!30 = !DILocation(line: 18, column: 14, scope: !29)
!31 = !DILocation(line: 0, scope: !21)
!32 = !DILocation(line: 20, column: 9, scope: !33)
!33 = distinct !DILexicalBlock(scope: !7, file: !1, line: 20, column: 7)
!34 = !DILocation(line: 20, column: 7, scope: !7)
!35 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 28, type: !36, scopeLine: 29, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !38)
!36 = !DISubroutineType(types: !37)
!37 = !{!10}
!38 = !{!39, !40, !41}
!39 = !DILocalVariable(name: "l", scope: !35, file: !1, line: 30, type: !10)
!40 = !DILocalVariable(name: "k", scope: !35, file: !1, line: 30, type: !10)
!41 = !DILocalVariable(name: "res", scope: !35, file: !1, line: 31, type: !10)
!42 = !DILocation(line: 30, column: 3, scope: !35)
