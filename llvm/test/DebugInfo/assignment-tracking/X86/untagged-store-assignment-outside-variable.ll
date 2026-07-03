; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG_

;; Generated from following C source that contains UB (read and write to
;; out of bounds static array element.
;; int a;
;; void b() {
;;   int c[2] = {0, 0};
;;   __attribute__((nodebug)) unsigned d = -1;
;;   if (a)
;;     c[a] = c[d] &= a;
;;   b();
;; }
;;
;; $ clang -O1 -g test.c -emit-llvm -S -o -
;;
;; Check the assignment c[d] isn't tracked (--implicit-check-not and
;; no assertion triggered, see llvm.org/PR65004).

; CHECK: bb.1.tailrecurse:
; CHECK: DBG_VALUE $noreg, $noreg, !18, !DIExpression()
; CHECK: DBG_VALUE %stack.0.c, $noreg, !18, !DIExpression(DW_OP_deref)
; CHECK: bb.2.if.then:

target triple = "x86_64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

define dso_local void @b() local_unnamed_addr !dbg !14 {
entry:
  %c = alloca [2 x i32], align 8, !DIAssignID !22
  br label %tailrecurse, !dbg !23

tailrecurse:                                      ; preds = %if.end, %entry
  call void @llvm.dbg.assign(metadata i1 undef, metadata !18, metadata !DIExpression(), metadata !22, metadata ptr %c, metadata !DIExpression()), !dbg !24
  store i64 0, ptr %c, align 8, !dbg !26, !DIAssignID !27
  call void @llvm.dbg.assign(metadata i64 0, metadata !18, metadata !DIExpression(), metadata !27, metadata ptr %c, metadata !DIExpression()), !dbg !24
  %0 = load i32, ptr @a, align 4, !dbg !28
  %tobool.not = icmp eq i32 %0, 0, !dbg !28
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !34

if.then:                                          ; preds = %tailrecurse
  %arrayidx = getelementptr inbounds [2 x i32], ptr %c, i64 0, i64 4294967295, !dbg !35
  %1 = load i32, ptr %arrayidx, align 4, !dbg !36
  %and = and i32 %1, %0, !dbg !36
  store i32 %and, ptr %arrayidx, align 4, !dbg !36
  %idxprom1 = sext i32 %0 to i64, !dbg !37
  %arrayidx2 = getelementptr inbounds [2 x i32], ptr %c, i64 0, i64 %idxprom1, !dbg !37
  store i32 %and, ptr %arrayidx2, align 4, !dbg !38
  br label %if.end, !dbg !37

if.end:                                           ; preds = %if.then, %tailrecurse
  br label %tailrecurse, !dbg !23
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 8, !"PIC Level", i32 2}
!10 = !{i32 7, !"PIE Level", i32 2}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!13 = !{!"clang version 17.0.0"}
!14 = distinct !DISubprogram(name: "b", scope: !3, file: !3, line: 2, type: !15, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18}
!18 = !DILocalVariable(name: "c", scope: !14, file: !3, line: 3, type: !19)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 64, elements: !20)
!20 = !{!21}
!21 = !DISubrange(count: 2)
!22 = distinct !DIAssignID()
!23 = !DILocation(line: 7, column: 3, scope: !14)
!24 = !DILocation(line: 0, scope: !14)
!25 = !DILocation(line: 3, column: 3, scope: !14)
!26 = !DILocation(line: 3, column: 7, scope: !14)
!27 = distinct !DIAssignID()
!28 = !DILocation(line: 5, column: 7, scope: !29)
!29 = distinct !DILexicalBlock(scope: !14, file: !3, line: 5, column: 7)
!34 = !DILocation(line: 5, column: 7, scope: !14)
!35 = !DILocation(line: 6, column: 12, scope: !29)
!36 = !DILocation(line: 6, column: 17, scope: !29)
!37 = !DILocation(line: 6, column: 5, scope: !29)
!38 = !DILocation(line: 6, column: 10, scope: !29)
!39 = !DILocation(line: 8, column: 1, scope: !14)
