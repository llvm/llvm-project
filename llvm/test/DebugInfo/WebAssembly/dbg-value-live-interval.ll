; RUN: llc < %s -print-machineinstrs 2>&1 | FileCheck %s

; CHECK: After WebAssembly Optimize Live Intervals:
; CHECK: bb.3.for.body.for.body_crit_edge:
; CHECK: [[REG:%[0-9]+]]:i32 = nsw ADD_I32 {{.*}} fib.c:7:7
; CHECK: DBG_VALUE [[REG]]:i32, $noreg, !"a", {{.*}} fib.c:5:13
; CHECK: After WebAssembly Memory Intrinsic Results:

; ModuleID = 'fib.bc'
source_filename = "fib.c"
; void swap(int* a, int* b);
;
; __attribute__ ((visibility ("default")))
; int fib(int n) {
;   int i, t, a = 0, b = 1;
;   for (i = 0; i < n; i++) {
;     a += b;
;     swap(&a, &b);
;   }
;   return b;
; }
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; Function Attrs: nounwind
define i32 @fib(i32 %n) local_unnamed_addr #0 !dbg !7 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %0 = bitcast i32* %a to i8*, !dbg !18
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !19
  store i32 0, i32* %a, align 4, !dbg !19, !tbaa !20
  %1 = bitcast i32* %b to i8*, !dbg !18
  store i32 1, i32* %b, align 4, !dbg !24, !tbaa !20
  %cmp4 = icmp sgt i32 %n, 0, !dbg !26
  br i1 %cmp4, label %for.body.preheader, label %for.end, !dbg !29

for.body.preheader:                               ; preds = %entry
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !19
  store i32 1, i32* %a, align 4, !dbg !30, !tbaa !20
  call void @llvm.dbg.value(metadata i32* %a, metadata !15, metadata !DIExpression()), !dbg !19
  call void @swap(i32* nonnull %a, i32* nonnull %b) #5, !dbg !32
  %2 = load i32, i32* %b, align 4, !dbg !33, !tbaa !20
  %exitcond9 = icmp eq i32 %n, 1, !dbg !26
  br i1 %exitcond9, label %for.end, label %for.body.for.body_crit_edge, !dbg !29, !llvm.loop !34

for.body.for.body_crit_edge:                      ; preds = %for.body.preheader, %for.body.for.body_crit_edge
  %3 = phi i32 [ %4, %for.body.for.body_crit_edge ], [ %2, %for.body.preheader ]
  %inc10 = phi i32 [ %inc, %for.body.for.body_crit_edge ], [ 1, %for.body.preheader ]
  %.pre = load i32, i32* %a, align 4, !dbg !30, !tbaa !20
  call void @llvm.dbg.value(metadata i32 %.pre, metadata !15, metadata !DIExpression()), !dbg !19
  %add = add nsw i32 %.pre, %3, !dbg !30
  call void @llvm.dbg.value(metadata i32 %add, metadata !15, metadata !DIExpression()), !dbg !19
  store i32 %add, i32* %a, align 4, !dbg !30, !tbaa !20
  call void @llvm.dbg.value(metadata i32* %a, metadata !15, metadata !DIExpression()), !dbg !19
  call void @swap(i32* nonnull %a, i32* nonnull %b) #5, !dbg !32
  %inc = add nuw nsw i32 %inc10, 1, !dbg !36
  %4 = load i32, i32* %b, align 4, !dbg !33, !tbaa !20
  %exitcond = icmp eq i32 %inc, %n, !dbg !26
  br i1 %exitcond, label %for.end, label %for.body.for.body_crit_edge, !dbg !29, !llvm.loop !34

for.end:                                          ; preds = %for.body.for.body_crit_edge, %for.body.preheader, %entry
  %.lcssa = phi i32 [ 1, %entry ], [ %2, %for.body.preheader ], [ %4, %for.body.for.body_crit_edge ]
  ret i32 %.lcssa, !dbg !38
}

declare void @swap(i32*, i32*) local_unnamed_addr #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 334610)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "fib.c", directory: "/d/y/llvmwasm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 334610)"}
!7 = distinct !DISubprogram(name: "fib", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!15}
!15 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 5, type: !10)
!17 = !DILocation(line: 4, column: 13, scope: !7)
!18 = !DILocation(line: 5, column: 3, scope: !7)
!19 = !DILocation(line: 5, column: 13, scope: !7)
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 5, column: 20, scope: !7)
!25 = !DILocation(line: 5, column: 7, scope: !7)
!26 = !DILocation(line: 6, column: 17, scope: !27)
!27 = distinct !DILexicalBlock(scope: !28, file: !1, line: 6, column: 3)
!28 = distinct !DILexicalBlock(scope: !7, file: !1, line: 6, column: 3)
!29 = !DILocation(line: 6, column: 3, scope: !28)
!30 = !DILocation(line: 7, column: 7, scope: !31)
!31 = distinct !DILexicalBlock(scope: !27, file: !1, line: 6, column: 27)
!32 = !DILocation(line: 8, column: 5, scope: !31)
!33 = !DILocation(line: 0, scope: !7)
!34 = distinct !{!34, !29, !35}
!35 = !DILocation(line: 9, column: 3, scope: !28)
!36 = !DILocation(line: 6, column: 23, scope: !27)
!37 = !DILocation(line: 11, column: 1, scope: !7)
!38 = !DILocation(line: 10, column: 3, scope: !7)
