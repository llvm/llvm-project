; RUN: opt %s -passes=loop-deletion -S -o - \
; RUN: | FileCheck %s

;; $ cat test.cpp:
;; void esc(int*);
;; void fun() {
;;   int Counter = 0;
;;   for (; Counter < 2; Counter++) {}
;;   esc(&Counter);
;; }
;;
;; IR grabbed before loop-deletion in:
;; $ clang++ -O2 -g -Xclang -fexperimental-assignment-tracking
;;
;; for.cond is a dead loop - the debug intrinsic inside will be moved to the
;; exit block (and made undef) and then the loop will be deleted. Ensure that
;; the dbg.assign intrinsic doesn't get transformed into a dbg.value by
;; mistake.

; CHECK: for.end:
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 poison,{{.+}}, metadata !DIExpression({{.+}}), metadata ![[ID:[0-9]+]], metadata ptr %Counter, metadata !DIExpression())
; CHECK-NEXT: store i32 2, ptr %Counter, align 4,{{.*}}!DIAssignID ![[ID]]

define dso_local void @_Z3funv() local_unnamed_addr #0 !dbg !7 {
entry:
  %Counter = alloca i32, align 4, !DIAssignID !13
  call void @llvm.dbg.assign(metadata i1 undef, metadata !11, metadata !DIExpression(), metadata !13, metadata ptr %Counter, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.assign(metadata i32 0, metadata !11, metadata !DIExpression(), metadata !16, metadata ptr %Counter, metadata !DIExpression()), !dbg !14
  br label %for.cond, !dbg !17

for.cond:                                         ; preds = %for.cond, %entry
  call void @llvm.dbg.assign(metadata i32 undef, metadata !11, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value), metadata !16, metadata ptr %Counter, metadata !DIExpression()), !dbg !14
  br i1 false, label %for.cond, label %for.end, !dbg !18, !llvm.loop !20

for.end:                                          ; preds = %for.cond
  store i32 2, ptr %Counter, align 4, !dbg !14, !DIAssignID !16
  call void @_Z3escPi(ptr noundef nonnull %Counter), !dbg !27
  ret void, !dbg !28
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, ptr nocapture)
declare !dbg !29 dso_local void @_Z3escPi(ptr noundef) local_unnamed_addr
declare void @llvm.lifetime.end.p0i8(i64 immarg, ptr nocapture)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0)"}
!7 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "Counter", scope: !7, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocation(line: 3, column: 3, scope: !7)
!16 = distinct !DIAssignID()
!17 = !DILocation(line: 4, column: 3, scope: !7)
!18 = !DILocation(line: 4, column: 3, scope: !19)
!19 = distinct !DILexicalBlock(scope: !7, file: !1, line: 4, column: 3)
!20 = distinct !{!20, !18, !21, !22}
!21 = !DILocation(line: 4, column: 35, scope: !19)
!22 = !{!"llvm.loop.mustprogress"}
!27 = !DILocation(line: 5, column: 3, scope: !7)
!28 = !DILocation(line: 6, column: 1, scope: !7)
!29 = !DISubprogram(name: "esc", linkageName: "_Z3escPi", scope: !1, file: !1, line: 1, type: !30, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !33)
!30 = !DISubroutineType(types: !31)
!31 = !{null, !32}
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!33 = !{}
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
