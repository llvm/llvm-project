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

; __attribute__((noinline))
; int f()
; {
;   int l, k;
;   fn3(&l);
;   fn3(&k);
;   fn1 (l, k);
;   return 0;
; }

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fn1(i32 noundef %0, i32 noundef %1) !dbg !10 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
    #dbg_declare(ptr %3, !15, !DIExpression(), !16)
  store i32 %1, ptr %4, align 4
    #dbg_declare(ptr %4, !17, !DIExpression(), !18)
    #dbg_declare(ptr %5, !19, !DIExpression(), !20)
  %7 = load i32, ptr %3, align 4, !dbg !21
  %8 = load i32, ptr %4, align 4, !dbg !22
  %9 = add nsw i32 %7, %8, !dbg !23
  store i32 %9, ptr %5, align 4, !dbg !20
  %10 = load i32, ptr %3, align 4, !dbg !24
  %11 = icmp sgt i32 %10, 1, !dbg !26
  br i1 %11, label %12, label %15, !dbg !27

12:                                               ; preds = %2
  %13 = load i32, ptr %5, align 4, !dbg !28
  %14 = add nsw i32 %13, 1, !dbg !28
  store i32 %14, ptr %5, align 4, !dbg !28
  br label %18, !dbg !29

15:                                               ; preds = %2
  %16 = load i32, ptr %5, align 4, !dbg !30
  %17 = add nsw i32 %16, 2, !dbg !30
  store i32 %17, ptr %5, align 4, !dbg !30
  br label %18

18:                                               ; preds = %15, %12
  %19 = load i32, ptr %4, align 4, !dbg !31
  %20 = icmp sgt i32 %19, 4, !dbg !33
  br i1 %20, label %21, label %25, !dbg !34

21:                                               ; preds = %18
  %22 = load i32, ptr %3, align 4, !dbg !35
  %23 = load i32, ptr %5, align 4, !dbg !36
  %24 = add nsw i32 %23, %22, !dbg !36
  store i32 %24, ptr %5, align 4, !dbg !36
  br label %25, !dbg !37

25:                                               ; preds = %21, %18
    #dbg_declare(ptr %6, !38, !DIExpression(), !39)
  store i32 7, ptr %6, align 4, !dbg !39
  %26 = load i32, ptr %6, align 4, !dbg !40
  call void @fn2(i32 noundef %26), !dbg !41
  %27 = load i32, ptr %5, align 4, !dbg !42
  %28 = add nsw i32 %27, -1, !dbg !42
  store i32 %28, ptr %5, align 4, !dbg !42
  ret void, !dbg !43
}

declare void @fn2(i32 noundef)

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f() !dbg !44 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
    #dbg_declare(ptr %1, !47, !DIExpression(), !48)
    #dbg_declare(ptr %2, !49, !DIExpression(), !50)
  call void @fn3(ptr noundef %1), !dbg !51
  call void @fn3(ptr noundef %2), !dbg !52
  %3 = load i32, ptr %1, align 4, !dbg !53
  %4 = load i32, ptr %2, align 4, !dbg !54
  call void @fn1(i32 noundef %3, i32 noundef %4), !dbg !55
  ret i32 0, !dbg !56
}

declare void @fn3(ptr noundef)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.1.7", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 19.1.7"}
!10 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 5, type: !11, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "x", arg: 1, scope: !10, file: !1, line: 5, type: !13)
!16 = !DILocation(line: 5, column: 10, scope: !10)
!17 = !DILocalVariable(name: "y", arg: 2, scope: !10, file: !1, line: 5, type: !13)
!18 = !DILocation(line: 5, column: 17, scope: !10)
!19 = !DILocalVariable(name: "u", scope: !10, file: !1, line: 7, type: !13)
!20 = !DILocation(line: 7, column: 7, scope: !10)
!21 = !DILocation(line: 7, column: 11, scope: !10)
!22 = !DILocation(line: 7, column: 15, scope: !10)
!23 = !DILocation(line: 7, column: 13, scope: !10)
!24 = !DILocation(line: 8, column: 7, scope: !25)
!25 = distinct !DILexicalBlock(scope: !10, file: !1, line: 8, column: 7)
!26 = !DILocation(line: 8, column: 9, scope: !25)
!27 = !DILocation(line: 8, column: 7, scope: !10)
!28 = !DILocation(line: 9, column: 7, scope: !25)
!29 = !DILocation(line: 9, column: 5, scope: !25)
!30 = !DILocation(line: 11, column: 7, scope: !25)
!31 = !DILocation(line: 12, column: 7, scope: !32)
!32 = distinct !DILexicalBlock(scope: !10, file: !1, line: 12, column: 7)
!33 = !DILocation(line: 12, column: 9, scope: !32)
!34 = !DILocation(line: 12, column: 7, scope: !10)
!35 = !DILocation(line: 13, column: 10, scope: !32)
!36 = !DILocation(line: 13, column: 7, scope: !32)
!37 = !DILocation(line: 13, column: 5, scope: !32)
!38 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 14, type: !13)
!39 = !DILocation(line: 14, column: 7, scope: !10)
!40 = !DILocation(line: 15, column: 8, scope: !10)
!41 = !DILocation(line: 15, column: 3, scope: !10)
!42 = !DILocation(line: 16, column: 5, scope: !10)
!43 = !DILocation(line: 17, column: 1, scope: !10)
!44 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 20, type: !45, scopeLine: 21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!45 = !DISubroutineType(types: !46)
!46 = !{!13}
!47 = !DILocalVariable(name: "l", scope: !44, file: !1, line: 22, type: !13)
!48 = !DILocation(line: 22, column: 7, scope: !44)
!49 = !DILocalVariable(name: "k", scope: !44, file: !1, line: 22, type: !13)
!50 = !DILocation(line: 22, column: 10, scope: !44)
!51 = !DILocation(line: 23, column: 3, scope: !44)
!52 = !DILocation(line: 24, column: 3, scope: !44)
!53 = !DILocation(line: 25, column: 8, scope: !44)
!54 = !DILocation(line: 25, column: 11, scope: !44)
!55 = !DILocation(line: 25, column: 3, scope: !44)
!56 = !DILocation(line: 26, column: 3, scope: !44)
