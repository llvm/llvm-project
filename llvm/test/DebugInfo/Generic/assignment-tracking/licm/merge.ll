; RUN: opt -passes=licm %s -S | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=licm %s -S | FileCheck %s

;; Ensure that we correctly merge the DIAssignID's from the sunk stores, add it
;; to the new new store instruction, and update the dbg.assign intrinsics using
;; them to use it instead.

;; Generated from the following, with some changes to the IR by hand:
;; $ cat test.c
;; void b(int c) {
;;   esc(&c);
;;   for (; c;  c++) // NOTE: I've added another store to c in the loop by hand.
;;   ;
;; }
;; $ clang -O2 -Xclang -disable-llvm-passes -g -emit-llvm -S -o a.ll
;; $ opt -passes=declare-to-assign,sroa,instcombine,simplifycfg,loop-simplify,lcssa,loop-rotate a.ll

; CHECK-LABEL: for.inc:
;; Check that the stores have actually been removed from this block, otherwise
;; this test is useless.
; CHECK-NOT: store i32 %inc, ptr %c.addr
;; Check that the two dbg.assigns now have the same (merged) !DIAssingID ID.
; CHECK: #dbg_assign(i32 %inc, ![[VAR_C:[0-9]+]], !DIExpression(), ![[ID:[0-9]+]], ptr %c.addr, !DIExpression(),
; CHECK-NOT: store i32 %inc, ptr %c.addr
; CHECK: #dbg_assign(i32 %inc, ![[VAR_C]], !DIExpression(), ![[ID]], ptr %c.addr, !DIExpression(),

; CHECK-LABEL: for.cond.for.end_crit_edge:
; CHECK-NEXT: %[[PHI:.*]] = phi i32 [ %inc, %for.inc ]
; CHECK-NEXT: store i32 %[[PHI]], ptr %c.addr{{.*}}, !DIAssignID ![[ID]]
; CHECK-NOT:  {{.*}}llvm.dbg{{.*}}
; CHECK-NEXT: br label %for.end

; CHECK: ![[VAR_C]] = !DILocalVariable(name: "c",

define dso_local void @b(i32 %c) !dbg !7 {
entry:
  %c.addr = alloca i32, align 4
  store i32 %c, ptr %c.addr, align 4, !DIAssignID !36
  call void @llvm.dbg.assign(metadata i32 %c, metadata !12, metadata !DIExpression(), metadata !36, metadata ptr %c.addr, metadata !DIExpression()), !dbg !13
  call void @esc(ptr nonnull %c.addr), !dbg !18
  %0 = load i32, ptr %c.addr, align 4, !dbg !19
  %tobool.not1 = icmp eq i32 %0, 0, !dbg !22
  br i1 %tobool.not1, label %for.end, label %for.inc.lr.ph, !dbg !22

for.inc.lr.ph:                                    ; preds = %entry
  br label %for.inc, !dbg !22

for.inc:                                          ; preds = %for.inc.lr.ph, %for.inc
  %1 = load i32, ptr %c.addr, align 4, !dbg !23
  %inc = add nsw i32 %1, 1, !dbg !23
  store i32 %inc, ptr %c.addr, align 4, !dbg !23, !DIAssignID !38
  call void @llvm.dbg.assign(metadata i32 %inc, metadata !12, metadata !DIExpression(), metadata !38, metadata ptr %c.addr, metadata !DIExpression()), !dbg !13
  ;; The following store and dbg.assign intrinsics are copies of those above,
  ;; with a new DIAssignID.
  store i32 %inc, ptr %c.addr, align 4, !dbg !23, !DIAssignID !37
  call void @llvm.dbg.assign(metadata i32 %inc, metadata !12, metadata !DIExpression(), metadata !37, metadata ptr %c.addr, metadata !DIExpression()), !dbg !13
  %2 = load i32, ptr %c.addr, align 4, !dbg !19
  %tobool.not = icmp eq i32 %2, 0, !dbg !22
  br i1 %tobool.not, label %for.cond.for.end_crit_edge, label %for.inc, !dbg !22, !llvm.loop !24

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  br label %for.end, !dbg !22

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void, !dbg !27
}


declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)
declare !dbg !28 dso_local void @esc(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "b", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "c", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 3, column: 3, scope: !7)
!19 = !DILocation(line: 4, column: 10, scope: !20)
!20 = distinct !DILexicalBlock(scope: !21, file: !1, line: 4, column: 3)
!21 = distinct !DILexicalBlock(scope: !7, file: !1, line: 4, column: 3)
!22 = !DILocation(line: 4, column: 3, scope: !21)
!23 = !DILocation(line: 4, column: 15, scope: !20)
!24 = distinct !{!24, !22, !25, !26}
!25 = !DILocation(line: 5, column: 3, scope: !21)
!26 = !{!"llvm.loop.mustprogress"}
!27 = !DILocation(line: 6, column: 1, scope: !7)
!28 = !DISubprogram(name: "esc", scope: !1, file: !1, line: 1, type: !29, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!36 = distinct !DIAssignID()
!37 = distinct !DIAssignID()
!38 = distinct !DIAssignID()
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
