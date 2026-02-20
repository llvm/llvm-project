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

; Function Attrs: noinline nounwind uwtable
define dso_local void @fn1(i32 noundef %0, i32 noundef %1) local_unnamed_addr !dbg !10 {
    #dbg_value(i32 %0, !15, !DIExpression(), !20)
    #dbg_value(i32 %1, !16, !DIExpression(), !20)
    #dbg_value(i32 poison, !17, !DIExpression(), !20)
  %3 = icmp sgt i32 %0, 1, !dbg !21
  %4 = select i1 %3, i32 1, i32 2, !dbg !23
    #dbg_value(i32 poison, !17, !DIExpression(), !20)
  %5 = icmp sgt i32 %1, 4, !dbg !24
  %6 = select i1 %5, i32 %0, i32 0, !dbg !26
    #dbg_value(i32 poison, !17, !DIExpression(), !20)
    #dbg_value(i32 7, !18, !DIExpression(), !20)
  tail call void @fn2(i32 noundef 7), !dbg !27
    #dbg_value(i32 poison, !19, !DIExpression(), !20)
    #dbg_value(i32 poison, !19, !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value), !20)
  %7 = add i32 %0, -1, !dbg !28
  %8 = add i32 %7, %1, !dbg !23
  %9 = add i32 %8, %4, !dbg !26
  %10 = add i32 %9, %6, !dbg !29
    #dbg_value(i32 %10, !17, !DIExpression(), !20)
  tail call void @fn2(i32 noundef %10), !dbg !30
  ret void, !dbg !31
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

declare !dbg !32 void @fn2(i32 noundef) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; Function Attrs: noinline nounwind uwtable
define dso_local noundef i32 @f() local_unnamed_addr !dbg !35 {
  %1 = alloca i32, align 4, !DIAssignID !41
    #dbg_assign(i1 poison, !39, !DIExpression(), !41, ptr %1, !DIExpression(), !42)
  %2 = alloca i32, align 4, !DIAssignID !43
    #dbg_assign(i1 poison, !40, !DIExpression(), !43, ptr %2, !DIExpression(), !42)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %1), !dbg !44
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %2), !dbg !44
  call void @fn3(ptr noundef nonnull %1), !dbg !45
  call void @fn3(ptr noundef nonnull %2), !dbg !46
  %3 = load i32, ptr %1, align 4, !dbg !47, !tbaa !48
  %4 = load i32, ptr %2, align 4, !dbg !52, !tbaa !48
  call void @fn1(i32 noundef %3, i32 noundef %4), !dbg !53
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %2), !dbg !54
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %1), !dbg !54
  ret i32 0, !dbg !55
}

declare !dbg !56 void @fn3(ptr noundef) local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.1.7", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 19.1.7"}
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
!23 = !DILocation(line: 5, column: 7, scope: !10)
!24 = !DILocation(line: 9, column: 9, scope: !25)
!25 = distinct !DILexicalBlock(scope: !10, file: !1, line: 9, column: 7)
!26 = !DILocation(line: 9, column: 7, scope: !10)
!27 = !DILocation(line: 12, column: 3, scope: !10)
!28 = !DILocation(line: 4, column: 13, scope: !10)
!29 = !DILocation(line: 15, column: 4, scope: !10)
!30 = !DILocation(line: 16, column: 3, scope: !10)
!31 = !DILocation(line: 17, column: 1, scope: !10)
!32 = !DISubprogram(name: "fn2", scope: !1, file: !1, line: 2, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !13}
!35 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 19, type: !36, scopeLine: 19, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !38)
!36 = !DISubroutineType(types: !37)
!37 = !{!13}
!38 = !{!39, !40}
!39 = !DILocalVariable(name: "l", scope: !35, file: !1, line: 20, type: !13)
!40 = !DILocalVariable(name: "k", scope: !35, file: !1, line: 20, type: !13)
!41 = distinct !DIAssignID()
!42 = !DILocation(line: 0, scope: !35)
!43 = distinct !DIAssignID()
!44 = !DILocation(line: 20, column: 3, scope: !35)
!45 = !DILocation(line: 21, column: 3, scope: !35)
!46 = !DILocation(line: 22, column: 3, scope: !35)
!47 = !DILocation(line: 23, column: 7, scope: !35)
!48 = !{!49, !49, i64 0}
!49 = !{!"int", !50, i64 0}
!50 = !{!"omnipotent char", !51, i64 0}
!51 = !{!"Simple C/C++ TBAA"}
!52 = !DILocation(line: 23, column: 10, scope: !35)
!53 = !DILocation(line: 23, column: 3, scope: !35)
!54 = !DILocation(line: 25, column: 1, scope: !35)
!55 = !DILocation(line: 24, column: 3, scope: !35)
!56 = !DISubprogram(name: "fn3", scope: !1, file: !1, line: 1, type: !57, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!57 = !DISubroutineType(types: !58)
!58 = !{null, !59}
!59 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
