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

; Function Attrs: noinline nounwind uwtable
define dso_local void @fn1(i32 noundef %0, i32 noundef %1) local_unnamed_addr !dbg !10 {
    #dbg_value(i32 %0, !15, !DIExpression(), !20)
    #dbg_value(i32 %1, !16, !DIExpression(), !20)
    #dbg_value(i32 poison, !17, !DIExpression(), !20)
  %3 = icmp sgt i32 %0, 1, !dbg !21
  %4 = select i1 %3, i32 1, i32 2, !dbg !21
    #dbg_value(i32 poison, !17, !DIExpression(), !20)
  %5 = icmp sgt i32 %1, 4, !dbg !23
  %6 = select i1 %5, i32 %0, i32 0, !dbg !23
    #dbg_value(i32 poison, !17, !DIExpression(), !20)
    #dbg_value(i32 7, !18, !DIExpression(), !20)
  tail call void @fn2(i32 noundef 7), !dbg !25
    #dbg_value(i32 poison, !19, !DIExpression(), !20)
    #dbg_value(i32 poison, !19, !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value), !20)
  %7 = add i32 %0, -1, !dbg !26
  %8 = add i32 %7, %1, !dbg !21
  %9 = add i32 %8, %4, !dbg !23
  %10 = add i32 %9, %6, !dbg !27
    #dbg_value(i32 %10, !17, !DIExpression(), !20)
  tail call void @fn2(i32 noundef %10), !dbg !28
  ret void, !dbg !29
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none))

declare !dbg !30 void @fn2(i32 noundef) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none))

; Function Attrs: noinline nounwind uwtable
define dso_local void @fn0(i32 noundef %0, i32 noundef %1) local_unnamed_addr !dbg !33 {
    #dbg_value(i32 %0, !35, !DIExpression(), !43)
    #dbg_value(i32 %1, !36, !DIExpression(), !43)
  %3 = add nsw i32 %1, %0, !dbg !44
    #dbg_value(i32 %3, !37, !DIExpression(), !43)
  %4 = icmp sgt i32 %1, 4, !dbg !45
  br i1 %4, label %5, label %7, !dbg !45

5:                                                ; preds = %2
  %6 = add nsw i32 %3, %0, !dbg !46
    #dbg_value(i32 %6, !37, !DIExpression(), !43)
    #dbg_value(i32 7, !38, !DIExpression(), !43)
  tail call void @fn2(i32 noundef 7), !dbg !48
  br label %7, !dbg !49

7:                                                ; preds = %2, %5
  %8 = phi i32 [ %6, %5 ], [ %3, %2 ], !dbg !43
    #dbg_value(i32 %8, !37, !DIExpression(), !43)
  %9 = add nsw i32 %8, -1, !dbg !50
    #dbg_value(i32 %9, !37, !DIExpression(), !43)
  tail call void @fn2(i32 noundef %9), !dbg !51
  ret void, !dbg !52
}

; Function Attrs: noinline nounwind uwtable
define dso_local noundef i32 @f() local_unnamed_addr !dbg !53 {
  %1 = alloca i32, align 4, !DIAssignID !59
    #dbg_assign(i1 poison, !57, !DIExpression(), !59, ptr %1, !DIExpression(), !60)
  %2 = alloca i32, align 4, !DIAssignID !61
    #dbg_assign(i1 poison, !58, !DIExpression(), !61, ptr %2, !DIExpression(), !60)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %1), !dbg !62
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %2), !dbg !62
  call void @fn3(ptr noundef nonnull %1), !dbg !63
  call void @fn3(ptr noundef nonnull %2), !dbg !64
  %3 = load i32, ptr %1, align 4, !dbg !65, !tbaa !66
  %4 = load i32, ptr %2, align 4, !dbg !70, !tbaa !66
  call void @fn1(i32 noundef %3, i32 noundef %4), !dbg !71
  %5 = load i32, ptr %1, align 4, !dbg !72, !tbaa !66
  %6 = load i32, ptr %2, align 4, !dbg !73, !tbaa !66
  call void @fn0(i32 noundef %5, i32 noundef %6), !dbg !74
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %2), !dbg !75
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %1), !dbg !75
  ret i32 0, !dbg !76
}

declare !dbg !77 void @fn3(ptr noundef) local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.1.8", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 21.1.8"}
!10 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15, !16, !17, !18, !19}
!15 = !DILocalVariable(name: "x", arg: 1, scope: !10, file: !1, line: 3, type: !13)
!16 = !DILocalVariable(name: "y", arg: 2, scope: !10, file: !1, line: 3, type: !13)
!17 = !DILocalVariable(name: "u", scope: !10, file: !1, line: 4, type: !13)
!18 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 11, type: !13)
!19 = !DILocalVariable(name: "v", scope: !10, file: !1, line: 13, type: !13)
!20 = !DILocation(line: 0, scope: !10)
!21 = !DILocation(line: 5, column: 9, scope: !22)
!22 = distinct !DILexicalBlock(scope: !10, file: !1, line: 5, column: 7)
!23 = !DILocation(line: 9, column: 9, scope: !24)
!24 = distinct !DILexicalBlock(scope: !10, file: !1, line: 9, column: 7)
!25 = !DILocation(line: 12, column: 3, scope: !10)
!26 = !DILocation(line: 4, column: 13, scope: !10)
!27 = !DILocation(line: 15, column: 4, scope: !10)
!28 = !DILocation(line: 16, column: 3, scope: !10)
!29 = !DILocation(line: 17, column: 1, scope: !10)
!30 = !DISubprogram(name: "fn2", scope: !1, file: !1, line: 2, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!31 = !DISubroutineType(types: !32)
!32 = !{null, !13}
!33 = distinct !DISubprogram(name: "fn0", scope: !1, file: !1, line: 19, type: !11, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !34)
!34 = !{!35, !36, !37, !38, !39}
!35 = !DILocalVariable(name: "x", arg: 1, scope: !33, file: !1, line: 19, type: !13)
!36 = !DILocalVariable(name: "y", arg: 2, scope: !33, file: !1, line: 19, type: !13)
!37 = !DILocalVariable(name: "u", scope: !33, file: !1, line: 20, type: !13)
!38 = !DILocalVariable(name: "a", scope: !33, file: !1, line: 21, type: !13)
!39 = !DILocalVariable(name: "v", scope: !40, file: !1, line: 27, type: !13)
!40 = distinct !DILexicalBlock(scope: !41, file: !1, line: 26, column: 21)
!41 = distinct !DILexicalBlock(scope: !42, file: !1, line: 26, column: 14)
!42 = distinct !DILexicalBlock(scope: !33, file: !1, line: 22, column: 7)
!43 = !DILocation(line: 0, scope: !33)
!44 = !DILocation(line: 20, column: 13, scope: !33)
!45 = !DILocation(line: 22, column: 9, scope: !42)
!46 = !DILocation(line: 23, column: 7, scope: !47)
!47 = distinct !DILexicalBlock(scope: !42, file: !1, line: 22, column: 14)
!48 = !DILocation(line: 25, column: 5, scope: !47)
!49 = !DILocation(line: 26, column: 3, scope: !47)
!50 = !DILocation(line: 30, column: 4, scope: !33)
!51 = !DILocation(line: 31, column: 3, scope: !33)
!52 = !DILocation(line: 32, column: 1, scope: !33)
!53 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 34, type: !54, scopeLine: 34, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !56)
!54 = !DISubroutineType(types: !55)
!55 = !{!13}
!56 = !{!57, !58}
!57 = !DILocalVariable(name: "l", scope: !53, file: !1, line: 35, type: !13)
!58 = !DILocalVariable(name: "k", scope: !53, file: !1, line: 35, type: !13)
!59 = distinct !DIAssignID()
!60 = !DILocation(line: 0, scope: !53)
!61 = distinct !DIAssignID()
!62 = !DILocation(line: 35, column: 3, scope: !53)
!63 = !DILocation(line: 36, column: 3, scope: !53)
!64 = !DILocation(line: 37, column: 3, scope: !53)
!65 = !DILocation(line: 38, column: 7, scope: !53)
!66 = !{!67, !67, i64 0}
!67 = !{!"int", !68, i64 0}
!68 = !{!"omnipotent char", !69, i64 0}
!69 = !{!"Simple C/C++ TBAA"}
!70 = !DILocation(line: 38, column: 10, scope: !53)
!71 = !DILocation(line: 38, column: 3, scope: !53)
!72 = !DILocation(line: 39, column: 7, scope: !53)
!73 = !DILocation(line: 39, column: 10, scope: !53)
!74 = !DILocation(line: 39, column: 3, scope: !53)
!75 = !DILocation(line: 41, column: 1, scope: !53)
!76 = !DILocation(line: 40, column: 3, scope: !53)
!77 = !DISubprogram(name: "fn3", scope: !1, file: !1, line: 1, type: !78, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!78 = !DISubroutineType(types: !79)
!79 = !{null, !80}
!80 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
