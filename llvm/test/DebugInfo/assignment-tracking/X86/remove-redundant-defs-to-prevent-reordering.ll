; RUN: llc %s -stop-before finalize-isel -o - \
; RUN:    -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --check-prefixes=CHECK,DBGVALUE --implicit-check-not="DBG_VALUE \$noreg"
; RUN: llc %s -stop-before finalize-isel -o - \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF --implicit-check-not="DBG_VALUE \$noreg"

;; Found in the wild, but this test involves modifications from:
;; int b;
;; void ext();
;; int fun(int a) {
;;   if (b == 0)
;;     ext();
;;
;;   a += b;
;;   return a;
;; }
;; A `dbg.assign(undef, ...)` has been added by hand in if.end.

;; For some variable we generate:
;;     %inc = add nuw nsw i32 %i.0128, 1
;;     call void @llvm.dbg.value(metadata i32 undef, ...
;;     call void @llvm.dbg.value(metadata i32 %inc, ...
;;
;; SelectionDAG swaps the dbg.value positions:
;;     %31:gr32 = nuw nsw ADD32ri8 %30:gr32(tied-def 0), 1
;;     DBG_VALUE %31:gr32, ...
;;     DBG_VALUE $noreg, ...
;;
;; Make sure to avoid this by removing redundant dbg.values after lowering
;; dbg.assigns.

;; Check that there's a debug instruction (--implicit-check-not checks no
;; `DBG_VALUE $noreg, ...` has been added).
; CHECK: bb.2.if.end:
; DBGVALUE: DBG_VALUE
; INSTRREF: INSTR_REF

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

; Function Attrs: uwtable mustprogress
define dso_local i32 @_Z3funi(i32 %a) local_unnamed_addr #0 !dbg !11 {
entry:
  call void @llvm.dbg.assign(metadata i1 undef, metadata !15, metadata !DIExpression(), metadata !16, metadata ptr undef, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.assign(metadata i32 %a, metadata !15, metadata !DIExpression(), metadata !18, metadata ptr undef, metadata !DIExpression()), !dbg !17
  %0 = load i32, ptr @b, align 4, !dbg !19
  %cmp = icmp eq i32 %0, 0, !dbg !25
  br i1 %cmp, label %if.then, label %if.end, !dbg !26

if.then:                                          ; preds = %entry
  tail call void @_Z3extv(), !dbg !27
  %.pre = load i32, ptr @b, align 4, !dbg !28
  br label %if.end, !dbg !27

if.end:                                           ; preds = %if.then, %entry
  %1 = phi i32 [ %.pre, %if.then ], [ %0, %entry ], !dbg !28
  %add = add nsw i32 %1, %a, !dbg !29
  ;; Added by hand:
  call void @llvm.dbg.assign(metadata i32 undef, metadata !15, metadata !DIExpression(), metadata !30, metadata ptr undef, metadata !DIExpression()), !dbg !17
call void @llvm.dbg.assign(metadata i32 %add, metadata !15, metadata !DIExpression(), metadata !30, metadata ptr undef, metadata !DIExpression()), !dbg !17
  ret i32 %add, !dbg !31
}

declare !dbg !32 dso_local void @_Z3extv() local_unnamed_addr #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !1000}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
!11 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funi", scope: !3, file: !3, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{!6, !6}
!14 = !{!15}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !11, file: !3, line: 3, type: !6)
!16 = distinct !DIAssignID()
!17 = !DILocation(line: 0, scope: !11)
!18 = distinct !DIAssignID()
!19 = !DILocation(line: 4, column: 7, scope: !20)
!20 = distinct !DILexicalBlock(scope: !11, file: !3, line: 4, column: 7)
!25 = !DILocation(line: 4, column: 9, scope: !20)
!26 = !DILocation(line: 4, column: 7, scope: !11)
!27 = !DILocation(line: 5, column: 5, scope: !20)
!28 = !DILocation(line: 7, column: 8, scope: !11)
!29 = !DILocation(line: 7, column: 5, scope: !11)
!30 = distinct !DIAssignID()
!31 = !DILocation(line: 8, column: 3, scope: !11)
!32 = !DISubprogram(name: "ext", linkageName: "_Z3extv", scope: !3, file: !3, line: 2, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!33 = !DISubroutineType(types: !34)
!34 = !{null}
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
