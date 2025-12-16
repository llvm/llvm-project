; RUN: opt -passes=loop-vectorize \
; RUN: -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -S < %s | FileCheck %s --check-prefix=IF-EVL

; RUN: opt -passes=loop-vectorize \
; RUN: -prefer-predicate-over-epilogue=scalar-epilogue \
; RUN: -mtriple=riscv64 -mattr=+v -S < %s | FileCheck %s --check-prefix=NO-VP

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"

define void @reverse_store(ptr %a, i64 %n) !dbg !4 {
entry:
    #dbg_value(ptr %a, !11, !DIExpression(), !15)
    #dbg_value(i64 %n, !12, !DIExpression(), !15)
    #dbg_value(i64 %n, !13, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value), !16)
  br label %for.body, !dbg !17

for.cond.cleanup:                                 ; preds = %for.body
  ret void, !dbg !18

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %n, %entry ], [ %indvars.iv.next, %for.body ]
    #dbg_value(i64 %indvars.iv, !13, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value), !16)
  %indvars.iv.next = add nsw i64 %indvars.iv, -1, !dbg !20
    #dbg_value(i64 %indvars.iv.next, !13, !DIExpression(), !16)
  %arrayidx = getelementptr inbounds nuw i32, ptr %a, i64 %indvars.iv.next, !dbg !21
  %1 = trunc nuw nsw i64 %indvars.iv.next to i32, !dbg !22
  store i32 %1, ptr %arrayidx, align 4, !dbg !22
    #dbg_value(i64 %indvars.iv.next, !13, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value), !16)
  %cmp = icmp samesign ugt i64 %indvars.iv, 1, !dbg !23
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !24, !llvm.loop !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dbg-tail-folding-by-evl.cpp", directory: "/test/file/path")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!4 = distinct !DISubprogram(name: "reverse_store", linkageName: "_Z13reverse_storePil", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !8}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!8 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11, !12, !13}
!11 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!12 = !DILocalVariable(name: "n", arg: 2, scope: !4, file: !1, line: 1, type: !8)
!13 = !DILocalVariable(name: "i", scope: !14, file: !1, line: 2, type: !9)
!14 = distinct !DILexicalBlock(scope: !4, file: !1, line: 2, column: 5)
!15 = !DILocation(line: 0, scope: !4)
!16 = !DILocation(line: 0, scope: !14)
!17 = !DILocation(line: 2, column: 5, scope: !14)
!18 = !DILocation(line: 4, column: 1, scope: !4)
!19 = distinct !DILexicalBlock(scope: !14, file: !1, line: 2, column: 5)
!20 = !DILocation(line: 2, scope: !14)
!21 = !DILocation(line: 3, column: 7, scope: !19)
!22 = !DILocation(line: 3, column: 12, scope: !19)
!23 = !DILocation(line: 2, column: 27, scope: !19)
!24 = !DILocation(line: 2, column: 5, scope: !14)
!25 = distinct !{!25, !17, !26, !27}
!26 = !DILocation(line: 3, column: 14, scope: !14)
!27 = !{!"llvm.loop.vectorize.enable", i1 true}
