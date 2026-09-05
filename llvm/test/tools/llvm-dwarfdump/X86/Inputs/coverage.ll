; The source code of the test case:
; extern void fn3(int *);
; extern void fn2(int);
; __attribute__((noinline)) void fn1(int x, int y) {
;   int u = x + y;
;   if (x > 1)
;     u += 1;
;   else
;     u += 2;
;   if (y > 4)
;     u += x;
;   int a = 7;
;   fn2(a);
;   int v = u;
;   v++;
;   u--;
;   fn2(u);
; }
; 
; __attribute__((noinline)) void fn0(int x, int y) {
;   int u = x + y;
;   int a;
;   if (y > 4) {
;     u += x;
;     a = 7;
;     fn2(a);
;   } else if (y > 5) {
;     int v = u;
;     v++;
;   }
;   u--;
;   fn2(u);
; }
; 
; __attribute__((noinline)) int f() {
;   int l, k;
;   fn3(&l);
;   fn3(&k);
;   fn1(l, k);
;   fn0(l, k);
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
  %7 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
    #dbg_declare(ptr %3, !15, !DIExpression(), !16)
  store i32 %1, ptr %4, align 4
    #dbg_declare(ptr %4, !17, !DIExpression(), !18)
    #dbg_declare(ptr %5, !19, !DIExpression(), !20)
  %8 = load i32, ptr %3, align 4, !dbg !21
  %9 = load i32, ptr %4, align 4, !dbg !22
  %10 = add nsw i32 %8, %9, !dbg !23
  store i32 %10, ptr %5, align 4, !dbg !20
  %11 = load i32, ptr %3, align 4, !dbg !24
  %12 = icmp sgt i32 %11, 1, !dbg !26
  br i1 %12, label %13, label %16, !dbg !26

13:                                               ; preds = %2
  %14 = load i32, ptr %5, align 4, !dbg !27
  %15 = add nsw i32 %14, 1, !dbg !27
  store i32 %15, ptr %5, align 4, !dbg !27
  br label %19, !dbg !28

16:                                               ; preds = %2
  %17 = load i32, ptr %5, align 4, !dbg !29
  %18 = add nsw i32 %17, 2, !dbg !29
  store i32 %18, ptr %5, align 4, !dbg !29
  br label %19

19:                                               ; preds = %16, %13
  %20 = load i32, ptr %4, align 4, !dbg !30
  %21 = icmp sgt i32 %20, 4, !dbg !32
  br i1 %21, label %22, label %26, !dbg !32

22:                                               ; preds = %19
  %23 = load i32, ptr %3, align 4, !dbg !33
  %24 = load i32, ptr %5, align 4, !dbg !34
  %25 = add nsw i32 %24, %23, !dbg !34
  store i32 %25, ptr %5, align 4, !dbg !34
  br label %26, !dbg !35

26:                                               ; preds = %22, %19
    #dbg_declare(ptr %6, !36, !DIExpression(), !37)
  store i32 7, ptr %6, align 4, !dbg !37
  %27 = load i32, ptr %6, align 4, !dbg !38
  call void @fn2(i32 noundef %27), !dbg !39
    #dbg_declare(ptr %7, !40, !DIExpression(), !41)
  %28 = load i32, ptr %5, align 4, !dbg !42
  store i32 %28, ptr %7, align 4, !dbg !41
  %29 = load i32, ptr %7, align 4, !dbg !43
  %30 = add nsw i32 %29, 1, !dbg !43
  store i32 %30, ptr %7, align 4, !dbg !43
  %31 = load i32, ptr %5, align 4, !dbg !44
  %32 = add nsw i32 %31, -1, !dbg !44
  store i32 %32, ptr %5, align 4, !dbg !44
  %33 = load i32, ptr %5, align 4, !dbg !45
  call void @fn2(i32 noundef %33), !dbg !46
  ret void, !dbg !47
}

declare void @fn2(i32 noundef)

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fn0(i32 noundef %0, i32 noundef %1) !dbg !48 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
    #dbg_declare(ptr %3, !49, !DIExpression(), !50)
  store i32 %1, ptr %4, align 4
    #dbg_declare(ptr %4, !51, !DIExpression(), !52)
    #dbg_declare(ptr %5, !53, !DIExpression(), !54)
  %8 = load i32, ptr %3, align 4, !dbg !55
  %9 = load i32, ptr %4, align 4, !dbg !56
  %10 = add nsw i32 %8, %9, !dbg !57
  store i32 %10, ptr %5, align 4, !dbg !54
    #dbg_declare(ptr %6, !58, !DIExpression(), !59)
  %11 = load i32, ptr %4, align 4, !dbg !60
  %12 = icmp sgt i32 %11, 4, !dbg !62
  br i1 %12, label %13, label %18, !dbg !62

13:                                               ; preds = %2
  %14 = load i32, ptr %3, align 4, !dbg !63
  %15 = load i32, ptr %5, align 4, !dbg !65
  %16 = add nsw i32 %15, %14, !dbg !65
  store i32 %16, ptr %5, align 4, !dbg !65
  store i32 7, ptr %6, align 4, !dbg !66
  %17 = load i32, ptr %6, align 4, !dbg !67
  call void @fn2(i32 noundef %17), !dbg !68
  br label %26, !dbg !69

18:                                               ; preds = %2
  %19 = load i32, ptr %4, align 4, !dbg !70
  %20 = icmp sgt i32 %19, 5, !dbg !72
  br i1 %20, label %21, label %25, !dbg !72

21:                                               ; preds = %18
    #dbg_declare(ptr %7, !73, !DIExpression(), !75)
  %22 = load i32, ptr %5, align 4, !dbg !76
  store i32 %22, ptr %7, align 4, !dbg !75
  %23 = load i32, ptr %7, align 4, !dbg !77
  %24 = add nsw i32 %23, 1, !dbg !77
  store i32 %24, ptr %7, align 4, !dbg !77
  br label %25, !dbg !78

25:                                               ; preds = %21, %18
  br label %26

26:                                               ; preds = %25, %13
  %27 = load i32, ptr %5, align 4, !dbg !79
  %28 = add nsw i32 %27, -1, !dbg !79
  store i32 %28, ptr %5, align 4, !dbg !79
  %29 = load i32, ptr %5, align 4, !dbg !80
  call void @fn2(i32 noundef %29), !dbg !81
  ret void, !dbg !82
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f() !dbg !83 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
    #dbg_declare(ptr %1, !86, !DIExpression(), !87)
    #dbg_declare(ptr %2, !88, !DIExpression(), !89)
  call void @fn3(ptr noundef %1), !dbg !90
  call void @fn3(ptr noundef %2), !dbg !91
  %3 = load i32, ptr %1, align 4, !dbg !92
  %4 = load i32, ptr %2, align 4, !dbg !93
  call void @fn1(i32 noundef %3, i32 noundef %4), !dbg !94
  %5 = load i32, ptr %1, align 4, !dbg !95
  %6 = load i32, ptr %2, align 4, !dbg !96
  call void @fn0(i32 noundef %5, i32 noundef %6), !dbg !97
  ret i32 0, !dbg !98
}

declare void @fn3(ptr noundef)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.1.8", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 21.1.8"}
!10 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "x", arg: 1, scope: !10, file: !1, line: 3, type: !13)
!16 = !DILocation(line: 3, column: 40, scope: !10)
!17 = !DILocalVariable(name: "y", arg: 2, scope: !10, file: !1, line: 3, type: !13)
!18 = !DILocation(line: 3, column: 47, scope: !10)
!19 = !DILocalVariable(name: "u", scope: !10, file: !1, line: 4, type: !13)
!20 = !DILocation(line: 4, column: 7, scope: !10)
!21 = !DILocation(line: 4, column: 11, scope: !10)
!22 = !DILocation(line: 4, column: 15, scope: !10)
!23 = !DILocation(line: 4, column: 13, scope: !10)
!24 = !DILocation(line: 5, column: 7, scope: !25)
!25 = distinct !DILexicalBlock(scope: !10, file: !1, line: 5, column: 7)
!26 = !DILocation(line: 5, column: 9, scope: !25)
!27 = !DILocation(line: 6, column: 7, scope: !25)
!28 = !DILocation(line: 6, column: 5, scope: !25)
!29 = !DILocation(line: 8, column: 7, scope: !25)
!30 = !DILocation(line: 9, column: 7, scope: !31)
!31 = distinct !DILexicalBlock(scope: !10, file: !1, line: 9, column: 7)
!32 = !DILocation(line: 9, column: 9, scope: !31)
!33 = !DILocation(line: 10, column: 10, scope: !31)
!34 = !DILocation(line: 10, column: 7, scope: !31)
!35 = !DILocation(line: 10, column: 5, scope: !31)
!36 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 11, type: !13)
!37 = !DILocation(line: 11, column: 7, scope: !10)
!38 = !DILocation(line: 12, column: 7, scope: !10)
!39 = !DILocation(line: 12, column: 3, scope: !10)
!40 = !DILocalVariable(name: "v", scope: !10, file: !1, line: 13, type: !13)
!41 = !DILocation(line: 13, column: 7, scope: !10)
!42 = !DILocation(line: 13, column: 11, scope: !10)
!43 = !DILocation(line: 14, column: 4, scope: !10)
!44 = !DILocation(line: 15, column: 4, scope: !10)
!45 = !DILocation(line: 16, column: 7, scope: !10)
!46 = !DILocation(line: 16, column: 3, scope: !10)
!47 = !DILocation(line: 17, column: 1, scope: !10)
!48 = distinct !DISubprogram(name: "fn0", scope: !1, file: !1, line: 19, type: !11, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!49 = !DILocalVariable(name: "x", arg: 1, scope: !48, file: !1, line: 19, type: !13)
!50 = !DILocation(line: 19, column: 40, scope: !48)
!51 = !DILocalVariable(name: "y", arg: 2, scope: !48, file: !1, line: 19, type: !13)
!52 = !DILocation(line: 19, column: 47, scope: !48)
!53 = !DILocalVariable(name: "u", scope: !48, file: !1, line: 20, type: !13)
!54 = !DILocation(line: 20, column: 7, scope: !48)
!55 = !DILocation(line: 20, column: 11, scope: !48)
!56 = !DILocation(line: 20, column: 15, scope: !48)
!57 = !DILocation(line: 20, column: 13, scope: !48)
!58 = !DILocalVariable(name: "a", scope: !48, file: !1, line: 21, type: !13)
!59 = !DILocation(line: 21, column: 7, scope: !48)
!60 = !DILocation(line: 22, column: 7, scope: !61)
!61 = distinct !DILexicalBlock(scope: !48, file: !1, line: 22, column: 7)
!62 = !DILocation(line: 22, column: 9, scope: !61)
!63 = !DILocation(line: 23, column: 10, scope: !64)
!64 = distinct !DILexicalBlock(scope: !61, file: !1, line: 22, column: 14)
!65 = !DILocation(line: 23, column: 7, scope: !64)
!66 = !DILocation(line: 24, column: 7, scope: !64)
!67 = !DILocation(line: 25, column: 9, scope: !64)
!68 = !DILocation(line: 25, column: 5, scope: !64)
!69 = !DILocation(line: 26, column: 3, scope: !64)
!70 = !DILocation(line: 26, column: 14, scope: !71)
!71 = distinct !DILexicalBlock(scope: !61, file: !1, line: 26, column: 14)
!72 = !DILocation(line: 26, column: 16, scope: !71)
!73 = !DILocalVariable(name: "v", scope: !74, file: !1, line: 27, type: !13)
!74 = distinct !DILexicalBlock(scope: !71, file: !1, line: 26, column: 21)
!75 = !DILocation(line: 27, column: 9, scope: !74)
!76 = !DILocation(line: 27, column: 13, scope: !74)
!77 = !DILocation(line: 28, column: 6, scope: !74)
!78 = !DILocation(line: 29, column: 3, scope: !74)
!79 = !DILocation(line: 30, column: 4, scope: !48)
!80 = !DILocation(line: 31, column: 7, scope: !48)
!81 = !DILocation(line: 31, column: 3, scope: !48)
!82 = !DILocation(line: 32, column: 1, scope: !48)
!83 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 34, type: !84, scopeLine: 34, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!84 = !DISubroutineType(types: !85)
!85 = !{!13}
!86 = !DILocalVariable(name: "l", scope: !83, file: !1, line: 35, type: !13)
!87 = !DILocation(line: 35, column: 7, scope: !83)
!88 = !DILocalVariable(name: "k", scope: !83, file: !1, line: 35, type: !13)
!89 = !DILocation(line: 35, column: 10, scope: !83)
!90 = !DILocation(line: 36, column: 3, scope: !83)
!91 = !DILocation(line: 37, column: 3, scope: !83)
!92 = !DILocation(line: 38, column: 7, scope: !83)
!93 = !DILocation(line: 38, column: 10, scope: !83)
!94 = !DILocation(line: 38, column: 3, scope: !83)
!95 = !DILocation(line: 39, column: 7, scope: !83)
!96 = !DILocation(line: 39, column: 10, scope: !83)
!97 = !DILocation(line: 39, column: 3, scope: !83)
!98 = !DILocation(line: 40, column: 3, scope: !83)
