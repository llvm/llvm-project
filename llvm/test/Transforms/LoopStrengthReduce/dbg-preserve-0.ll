; RUN: opt -loop-reduce -S %s | FileCheck %s

;; Test that LSR preserves debug-info for induction variables and scev-based
;; salvaging produces short DIExpressions that use a constant offset from the
;; induction variable.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define dso_local void @foo(ptr nocapture %p) local_unnamed_addr !dbg !7 {
; CHECK-LABEL: @foo(
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !13, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i8 0, metadata !14, metadata !DIExpression()), !dbg !17
  br label %for.body, !dbg !18

for.cond.cleanup:                                 ; preds = %for.body
  ret void, !dbg !19

for.body:                                         ; preds = %entry, %for.body
; CHECK-LABEL: for.body:
  %i.06 = phi i8 [ 0, %entry ], [ %inc, %for.body ]
  %p.addr.05 = phi ptr [ %p, %entry ], [ %add.ptr, %for.body ]
  call void @llvm.dbg.value(metadata i8 %i.06, metadata !14, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata ptr %p.addr.05, metadata !13, metadata !DIExpression()), !dbg !16
; CHECK-NOT: call void @llvm.dbg.value(metadata ptr undef
; CHECK: all void @llvm.dbg.value(metadata ptr %lsr.iv, metadata ![[MID_p:[0-9]+]],  metadata !DIExpression(DW_OP_constu, 3, DW_OP_minus, DW_OP_stack_value))
  %add.ptr = getelementptr inbounds i8, ptr %p.addr.05, i64 3, !dbg !20
  call void @llvm.dbg.value(metadata ptr %add.ptr, metadata !13, metadata !DIExpression()), !dbg !16
; CHECK-NOT: call void @llvm.dbg.value(metadata ptr undef
; CHECK: call void @llvm.dbg.value(metadata ptr %lsr.iv, metadata ![[MID_p]], metadata !DIExpression())
  store i8 %i.06, ptr %add.ptr, align 1, !dbg !23, !tbaa !24
  %inc = add nuw nsw i8 %i.06, 1, !dbg !27
  call void @llvm.dbg.value(metadata i8 %inc, metadata !14, metadata !DIExpression()), !dbg !17
  %exitcond.not = icmp eq i8 %inc, 32, !dbg !28
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !dbg !18, !llvm.loop !29
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "lsrdbg.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, line: 2, type: !10)
; CHECK: ![[MID_p]] = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!14 = !DILocalVariable(name: "i", scope: !15, file: !1, line: 4, type: !11)
!15 = distinct !DILexicalBlock(scope: !7, file: !1, line: 4, column: 3)
!16 = !DILocation(line: 0, scope: !7)
!17 = !DILocation(line: 0, scope: !15)
!18 = !DILocation(line: 4, column: 3, scope: !15)
!19 = !DILocation(line: 8, column: 1, scope: !7)
!20 = !DILocation(line: 5, column: 7, scope: !21)
!21 = distinct !DILexicalBlock(scope: !22, file: !1, line: 4, column: 42)
!22 = distinct !DILexicalBlock(scope: !15, file: !1, line: 4, column: 3)
!23 = !DILocation(line: 6, column: 8, scope: !21)
!24 = !{!25, !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 4, column: 38, scope: !22)
!28 = !DILocation(line: 4, column: 31, scope: !22)
!29 = distinct !{!29, !18, !30, !31}
!30 = !DILocation(line: 7, column: 3, scope: !15)
!31 = !{!"llvm.loop.unroll.disable"}
