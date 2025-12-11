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

; Function Attrs: noinline nounwind uwtable
define dso_local void @fn1(i32 %0, i32 %1) local_unnamed_addr !dbg !10 {
    #dbg_value(i32 poison, !15, !DIExpression(), !19)
    #dbg_value(i32 poison, !16, !DIExpression(), !19)
    #dbg_value(!DIArgList(i32 poison, i32 poison), !17, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !19)
    #dbg_value(i32 poison, !17, !DIExpression(), !19)
    #dbg_value(i32 poison, !17, !DIExpression(), !19)
    #dbg_value(i32 7, !18, !DIExpression(), !19)
  tail call void @fn2(i32 noundef 7) #3, !dbg !20
    #dbg_value(i32 poison, !17, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value), !19)
  ret void, !dbg !21
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

declare !dbg !22 void @fn2(i32 noundef) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; Function Attrs: noinline nounwind uwtable
define dso_local noundef i32 @f() local_unnamed_addr !dbg !25 {
  %1 = alloca i32, align 4, !DIAssignID !31
    #dbg_assign(i1 poison, !29, !DIExpression(), !31, ptr %1, !DIExpression(), !32)
  %2 = alloca i32, align 4, !DIAssignID !33
    #dbg_assign(i1 poison, !30, !DIExpression(), !33, ptr %2, !DIExpression(), !32)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %1) #3, !dbg !34
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %2) #3, !dbg !34
  call void @fn3(ptr noundef nonnull %1) #3, !dbg !35
  call void @fn3(ptr noundef nonnull %2) #3, !dbg !36
  call void @fn1(i32 poison, i32 poison), !dbg !37
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %2) #3, !dbg !38
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %1) #3, !dbg !38
  ret i32 0, !dbg !39
}

declare !dbg !40 void @fn3(ptr noundef) local_unnamed_addr

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
!10 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 5, type: !11, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15, !16, !17, !18}
!15 = !DILocalVariable(name: "x", arg: 1, scope: !10, file: !1, line: 5, type: !13)
!16 = !DILocalVariable(name: "y", arg: 2, scope: !10, file: !1, line: 5, type: !13)
!17 = !DILocalVariable(name: "u", scope: !10, file: !1, line: 7, type: !13)
!18 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 14, type: !13)
!19 = !DILocation(line: 0, scope: !10)
!20 = !DILocation(line: 15, column: 3, scope: !10)
!21 = !DILocation(line: 17, column: 1, scope: !10)
!22 = !DISubprogram(name: "fn2", scope: !1, file: !1, line: 2, type: !23, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !13}
!25 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 20, type: !26, scopeLine: 21, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !28)
!26 = !DISubroutineType(types: !27)
!27 = !{!13}
!28 = !{!29, !30}
!29 = !DILocalVariable(name: "l", scope: !25, file: !1, line: 22, type: !13)
!30 = !DILocalVariable(name: "k", scope: !25, file: !1, line: 22, type: !13)
!31 = distinct !DIAssignID()
!32 = !DILocation(line: 0, scope: !25)
!33 = distinct !DIAssignID()
!34 = !DILocation(line: 22, column: 3, scope: !25)
!35 = !DILocation(line: 23, column: 3, scope: !25)
!36 = !DILocation(line: 24, column: 3, scope: !25)
!37 = !DILocation(line: 25, column: 3, scope: !25)
!38 = !DILocation(line: 27, column: 1, scope: !25)
!39 = !DILocation(line: 26, column: 3, scope: !25)
!40 = !DISubprogram(name: "fn3", scope: !1, file: !1, line: 1, type: !41, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!41 = !DISubroutineType(types: !42)
!42 = !{null, !43}
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
