; FIXME: use -stop-after when MIR serialization output includes needed debug info.
; RUN: llc < %s -print-after=wasm-reg-stackify 2>&1 | FileCheck %s

; CHECK: {{.*}}After WebAssembly Register Stackify
; CHECK: bb.3.for.body.for.body_crit_edge:
; CHECK: [[REG:%[0-9]+]]:i32 = nsw ADD_I32 {{.*}} fib.c:7:7
; CHECK-NEXT: DBG_VALUE [[REG]]:i32, $noreg, !"a", {{.*}} fib.c:5:13

; ModuleID = 'fib.bc'
; The test generated via: clang --target=wasm32-unknown-unknown-wasm fib.c -g -O2
; All lifetime markers and attributes were removed.
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
  call void @llvm.dbg.value(metadata i32 %n, metadata !12, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !19
  store i32 0, ptr %a, align 4, !dbg !19, !tbaa !20
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !24
  store i32 1, ptr %b, align 4, !dbg !24, !tbaa !20
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !25
  %cmp4 = icmp sgt i32 %n, 0, !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !24
  br i1 %cmp4, label %for.body.preheader, label %for.end, !dbg !29

for.body.preheader:                               ; preds = %entry
  call void @llvm.dbg.value(metadata i32 0, metadata !13, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 1, metadata !15, metadata !DIExpression()), !dbg !19
  store i32 1, ptr %a, align 4, !dbg !30, !tbaa !20
  call void @llvm.dbg.value(metadata ptr %a, metadata !15, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata ptr %b, metadata !16, metadata !DIExpression()), !dbg !24
  call void @swap(ptr nonnull %a, ptr nonnull %b) #5, !dbg !32
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !25
  %0 = load i32, ptr %b, align 4, !dbg !33, !tbaa !20
  call void @llvm.dbg.value(metadata i32 %0, metadata !16, metadata !DIExpression()), !dbg !24
  %exitcond9 = icmp eq i32 %n, 1, !dbg !26
  br i1 %exitcond9, label %for.end, label %for.body.for.body_crit_edge, !dbg !29, !llvm.loop !34

for.body.for.body_crit_edge:                      ; preds = %for.body.preheader, %for.body.for.body_crit_edge
  %1 = phi i32 [ %2, %for.body.for.body_crit_edge ], [ %0, %for.body.preheader ]
  %inc10 = phi i32 [ %inc, %for.body.for.body_crit_edge ], [ 1, %for.body.preheader ]
  %.pre = load i32, ptr %a, align 4, !dbg !30, !tbaa !20
  call void @llvm.dbg.value(metadata i32 %inc10, metadata !13, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.value(metadata i32 %.pre, metadata !15, metadata !DIExpression()), !dbg !19
  %add = add nsw i32 %.pre, %1, !dbg !30
  call void @llvm.dbg.value(metadata i32 %add, metadata !15, metadata !DIExpression()), !dbg !19
  store i32 %add, ptr %a, align 4, !dbg !30, !tbaa !20
  call void @llvm.dbg.value(metadata ptr %a, metadata !15, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata ptr %b, metadata !16, metadata !DIExpression()), !dbg !24
  call void @swap(ptr nonnull %a, ptr nonnull %b) #5, !dbg !32
  %inc = add nuw nsw i32 %inc10, 1, !dbg !36
  call void @llvm.dbg.value(metadata i32 %inc, metadata !13, metadata !DIExpression()), !dbg !25
  %2 = load i32, ptr %b, align 4, !dbg !33, !tbaa !20
  call void @llvm.dbg.value(metadata i32 %2, metadata !16, metadata !DIExpression()), !dbg !24
  %exitcond = icmp eq i32 %inc, %n, !dbg !26
  br i1 %exitcond, label %for.end, label %for.body.for.body_crit_edge, !dbg !29, !llvm.loop !34

for.end:                                          ; preds = %for.body.for.body_crit_edge, %for.body.preheader, %entry
  %.lcssa = phi i32 [ 1, %entry ], [ %0, %for.body.preheader ], [ %2, %for.body.for.body_crit_edge ]
  ret i32 %.lcssa, !dbg !38
}

declare void @swap(ptr, ptr) local_unnamed_addr #2

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
!11 = !{!12, !13, !14, !15, !16}
!12 = !DILocalVariable(name: "n", arg: 1, scope: !7, file: !1, line: 4, type: !10)
!13 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 5, type: !10)
!14 = !DILocalVariable(name: "t", scope: !7, file: !1, line: 5, type: !10)
!15 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 5, type: !10)
!16 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 5, type: !10)
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
!39 = distinct !DISubprogram(name: "_start", scope: !1, file: !1, line: 13, type: !40, isLocal: false, isDefinition: true, scopeLine: 13, isOptimized: true, unit: !0, retainedNodes: !2)
!40 = !DISubroutineType(types: !41)
!41 = !{null}
!42 = !DILocation(line: 13, column: 16, scope: !39)
