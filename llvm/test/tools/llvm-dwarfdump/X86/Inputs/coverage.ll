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
; __attribute__((noinline)) int f() {
;   int l, k;
;   fn3(&l);
;   fn3(&k);
;   fn1(l, k);
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
  br i1 %12, label %13, label %16, !dbg !27

13:                                               ; preds = %2
  %14 = load i32, ptr %5, align 4, !dbg !28
  %15 = add nsw i32 %14, 1, !dbg !28
  store i32 %15, ptr %5, align 4, !dbg !28
  br label %19, !dbg !29

16:                                               ; preds = %2
  %17 = load i32, ptr %5, align 4, !dbg !30
  %18 = add nsw i32 %17, 2, !dbg !30
  store i32 %18, ptr %5, align 4, !dbg !30
  br label %19

19:                                               ; preds = %16, %13
  %20 = load i32, ptr %4, align 4, !dbg !31
  %21 = icmp sgt i32 %20, 4, !dbg !33
  br i1 %21, label %22, label %26, !dbg !34

22:                                               ; preds = %19
  %23 = load i32, ptr %3, align 4, !dbg !35
  %24 = load i32, ptr %5, align 4, !dbg !36
  %25 = add nsw i32 %24, %23, !dbg !36
  store i32 %25, ptr %5, align 4, !dbg !36
  br label %26, !dbg !37

26:                                               ; preds = %22, %19
    #dbg_declare(ptr %6, !38, !DIExpression(), !39)
  store i32 7, ptr %6, align 4, !dbg !39
  %27 = load i32, ptr %6, align 4, !dbg !40
  call void @fn2(i32 noundef %27), !dbg !41
    #dbg_declare(ptr %7, !42, !DIExpression(), !43)
  %28 = load i32, ptr %5, align 4, !dbg !44
  store i32 %28, ptr %7, align 4, !dbg !43
  %29 = load i32, ptr %7, align 4, !dbg !45
  %30 = add nsw i32 %29, 1, !dbg !45
  store i32 %30, ptr %7, align 4, !dbg !45
  %31 = load i32, ptr %5, align 4, !dbg !46
  %32 = add nsw i32 %31, -1, !dbg !46
  store i32 %32, ptr %5, align 4, !dbg !46
  %33 = load i32, ptr %5, align 4, !dbg !47
  call void @fn2(i32 noundef %33), !dbg !48
  ret void, !dbg !49
}

declare void @fn2(i32 noundef)

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @f() !dbg !50 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
    #dbg_declare(ptr %1, !53, !DIExpression(), !54)
    #dbg_declare(ptr %2, !55, !DIExpression(), !56)
  call void @fn3(ptr noundef %1), !dbg !57
  call void @fn3(ptr noundef %2), !dbg !58
  %3 = load i32, ptr %1, align 4, !dbg !59
  %4 = load i32, ptr %2, align 4, !dbg !60
  call void @fn1(i32 noundef %3, i32 noundef %4), !dbg !61
  ret i32 0, !dbg !62
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
!27 = !DILocation(line: 5, column: 7, scope: !10)
!28 = !DILocation(line: 6, column: 7, scope: !25)
!29 = !DILocation(line: 6, column: 5, scope: !25)
!30 = !DILocation(line: 8, column: 7, scope: !25)
!31 = !DILocation(line: 9, column: 7, scope: !32)
!32 = distinct !DILexicalBlock(scope: !10, file: !1, line: 9, column: 7)
!33 = !DILocation(line: 9, column: 9, scope: !32)
!34 = !DILocation(line: 9, column: 7, scope: !10)
!35 = !DILocation(line: 10, column: 10, scope: !32)
!36 = !DILocation(line: 10, column: 7, scope: !32)
!37 = !DILocation(line: 10, column: 5, scope: !32)
!38 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 11, type: !13)
!39 = !DILocation(line: 11, column: 7, scope: !10)
!40 = !DILocation(line: 12, column: 7, scope: !10)
!41 = !DILocation(line: 12, column: 3, scope: !10)
!42 = !DILocalVariable(name: "v", scope: !10, file: !1, line: 13, type: !13)
!43 = !DILocation(line: 13, column: 7, scope: !10)
!44 = !DILocation(line: 13, column: 11, scope: !10)
!45 = !DILocation(line: 14, column: 4, scope: !10)
!46 = !DILocation(line: 15, column: 4, scope: !10)
!47 = !DILocation(line: 16, column: 7, scope: !10)
!48 = !DILocation(line: 16, column: 3, scope: !10)
!49 = !DILocation(line: 17, column: 1, scope: !10)
!50 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 19, type: !51, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!51 = !DISubroutineType(types: !52)
!52 = !{!13}
!53 = !DILocalVariable(name: "l", scope: !50, file: !1, line: 20, type: !13)
!54 = !DILocation(line: 20, column: 7, scope: !50)
!55 = !DILocalVariable(name: "k", scope: !50, file: !1, line: 20, type: !13)
!56 = !DILocation(line: 20, column: 10, scope: !50)
!57 = !DILocation(line: 21, column: 3, scope: !50)
!58 = !DILocation(line: 22, column: 3, scope: !50)
!59 = !DILocation(line: 23, column: 7, scope: !50)
!60 = !DILocation(line: 23, column: 10, scope: !50)
!61 = !DILocation(line: 23, column: 3, scope: !50)
!62 = !DILocation(line: 24, column: 3, scope: !50)
