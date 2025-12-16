; RUN: opt -passes=loop-vectorize \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -S < %s | FileCheck %s --check-prefix=IF-EVL

; RUN: opt -passes=loop-vectorize \
; RUN: -prefer-predicate-over-epilogue=scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -S < %s | FileCheck %s --check-prefix=NO-VP

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"

define void @reverse_store(ptr %a, i32 %n) !dbg !4 {
entry:
    #dbg_value(ptr %a, !10, !DIExpression(), !14)
    #dbg_value(i32 %n, !11, !DIExpression(), !14)
    #dbg_value(i32 %n, !12, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value), !15)
  %cmp4 = icmp sgt i32 %n, 0, !dbg !16
  br i1 %cmp4, label %for.body.preheader, label %for.cond.cleanup, !dbg !18

for.body.preheader:                               ; preds = %entry
  %0 = zext nneg i32 %n to i64, !dbg !19
  br label %for.body, !dbg !19

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup, !dbg !20

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void, !dbg !20

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
    #dbg_value(i64 %indvars.iv, !12, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value), !15)
  %indvars.iv.next = add nsw i64 %indvars.iv, -1, !dbg !21
    #dbg_value(i64 %indvars.iv.next, !12, !DIExpression(), !15)
  %arrayidx = getelementptr inbounds nuw i32, ptr %a, i64 %indvars.iv.next, !dbg !22
  %1 = trunc nuw nsw i64 %indvars.iv.next to i32, !dbg !23
  store i32 %1, ptr %arrayidx, align 4, !dbg !23
    #dbg_value(i64 %indvars.iv.next, !12, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value), !15)
  %cmp = icmp samesign ugt i64 %indvars.iv, 1, !dbg !24
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit, !dbg !25, !llvm.loop !26
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dbg-tail-folding-by-evl.cpp", directory: "/test/file/path")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!4 = distinct !DISubprogram(name: "reverse_store", linkageName: "_Z13reverse_storePii", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !8}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10, !11, !12}
!10 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!11 = !DILocalVariable(name: "n", arg: 2, scope: !4, file: !1, line: 1, type: !8)
!12 = !DILocalVariable(name: "i", scope: !13, file: !1, line: 2, type: !8)
!13 = distinct !DILexicalBlock(scope: !4, file: !1, line: 2, column: 5)
!14 = !DILocation(line: 0, scope: !4)
!15 = !DILocation(line: 0, scope: !13)
!16 = !DILocation(line: 2, column: 27, scope: !17)
!17 = distinct !DILexicalBlock(scope: !13, file: !1, line: 2, column: 5)
!18 = !DILocation(line: 2, column: 5, scope: !13)
!19 = !DILocation(line: 2, column: 5, scope: !13)
!20 = !DILocation(line: 4, column: 1, scope: !4)
!21 = !DILocation(line: 2, scope: !13)
!22 = !DILocation(line: 3, column: 7, scope: !17)
!23 = !DILocation(line: 3, column: 12, scope: !17)
!24 = !DILocation(line: 2, column: 27, scope: !17)
!25 = !DILocation(line: 2, column: 5, scope: !13)
!26 = distinct !{!26, !19, !27, !28}
!27 = !DILocation(line: 3, column: 14, scope: !13)
!28 = !{!"llvm.loop.vectorize.enable", i1 true}
