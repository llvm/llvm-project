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
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local void @fn1(i32 noundef %0, i32 noundef %1) local_unnamed_addr !dbg !9 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %1, metadata !15, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata !DIArgList(i32 %0, i32 %1), metadata !16, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)), !dbg !18
  call void @llvm.dbg.value(metadata i32 undef, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 undef, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 7, metadata !17, metadata !DIExpression()), !dbg !18
  tail call void @fn2(i32 noundef 7), !dbg !19
  call void @llvm.dbg.value(metadata i32 undef, metadata !16, metadata !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !18
  ret void, !dbg !20
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare !dbg !21 void @fn2(i32 noundef) local_unnamed_addr

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @f() local_unnamed_addr !dbg !25 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = bitcast i32* %1 to i8*, !dbg !31
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %3), !dbg !31
  %4 = bitcast i32* %2 to i8*, !dbg !31
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %4), !dbg !31
  call void @llvm.dbg.value(metadata i32* %1, metadata !29, metadata !DIExpression(DW_OP_deref)), !dbg !32
  call void @fn3(i32* noundef nonnull %1), !dbg !33
  call void @llvm.dbg.value(metadata i32* %2, metadata !30, metadata !DIExpression(DW_OP_deref)), !dbg !32
  call void @fn3(i32* noundef nonnull %2), !dbg !34
  %5 = load i32, i32* %1, align 4, !dbg !35, !tbaa !36
  call void @llvm.dbg.value(metadata i32 %5, metadata !29, metadata !DIExpression()), !dbg !32
  %6 = load i32, i32* %2, align 4, !dbg !40, !tbaa !36
  call void @llvm.dbg.value(metadata i32 %6, metadata !30, metadata !DIExpression()), !dbg !32
  call void @fn1(i32 noundef %5, i32 noundef %6), !dbg !41
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %4), !dbg !42
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %3), !dbg !42
  ret i32 0, !dbg !43
}

declare !dbg !44 void @fn3(i32* noundef) local_unnamed_addr

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.6", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 1}
!8 = !{!"clang version 14.0.6"}
!9 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 5, type: !10, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!14, !15, !16, !17}
!14 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !1, line: 5, type: !12)
!15 = !DILocalVariable(name: "y", arg: 2, scope: !9, file: !1, line: 5, type: !12)
!16 = !DILocalVariable(name: "u", scope: !9, file: !1, line: 7, type: !12)
!17 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 14, type: !12)
!18 = !DILocation(line: 0, scope: !9)
!19 = !DILocation(line: 15, column: 3, scope: !9)
!20 = !DILocation(line: 17, column: 1, scope: !9)
!21 = !DISubprogram(name: "fn2", scope: !1, file: !1, line: 2, type: !22, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !24)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !12}
!24 = !{}
!25 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 20, type: !26, scopeLine: 21, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !28)
!26 = !DISubroutineType(types: !27)
!27 = !{!12}
!28 = !{!29, !30}
!29 = !DILocalVariable(name: "l", scope: !25, file: !1, line: 22, type: !12)
!30 = !DILocalVariable(name: "k", scope: !25, file: !1, line: 22, type: !12)
!31 = !DILocation(line: 22, column: 3, scope: !25)
!32 = !DILocation(line: 0, scope: !25)
!33 = !DILocation(line: 23, column: 3, scope: !25)
!34 = !DILocation(line: 24, column: 3, scope: !25)
!35 = !DILocation(line: 25, column: 8, scope: !25)
!36 = !{!37, !37, i64 0}
!37 = !{!"int", !38, i64 0}
!38 = !{!"omnipotent char", !39, i64 0}
!39 = !{!"Simple C/C++ TBAA"}
!40 = !DILocation(line: 25, column: 11, scope: !25)
!41 = !DILocation(line: 25, column: 3, scope: !25)
!42 = !DILocation(line: 27, column: 1, scope: !25)
!43 = !DILocation(line: 26, column: 3, scope: !25)
!44 = !DISubprogram(name: "fn3", scope: !1, file: !1, line: 1, type: !45, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !24)
!45 = !DISubroutineType(types: !46)
!46 = !{null, !47}
!47 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
