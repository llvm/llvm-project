; RUN: opt -S -loop-reduce %s -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -S -loop-reduce %s -o - | FileCheck %s
; REQUIRES: x86-registered-target

;; Ensure that we retain debuginfo for the induction variable and dependant
;; variables when loop strength reduction is applied to the loop.
;; This IR produced from:
;;
;; clang -S -emit-llvm -Xclang -disable-llvm-passes -g lsr-basic.cpp -o
;; Then executing opt -O2 up to the the loopFullUnroll pass.
;; void mul_pow_of_2_to_shift_var_inc(unsigned size, unsigned *data, unsigned multiplicand) {
;;     unsigned i = 0;
;; #pragma clang loop vectorize(disable)
;;     while (i < size) {
;;         unsigned comp = i * multiplicand;
;;         data[i] = comp; 
;;         i++;
;;     }
;; }
; CHECK: #dbg_value(i64 %lsr.iv, ![[i:[0-9]+]], !DIExpression(),
; CHECK: #dbg_value(!DIArgList(i64 %lsr.iv, i32 %multiplicand), ![[comp:[0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_mul, DW_OP_stack_value),
; CHECK: #dbg_value(i64 %lsr.iv, ![[i]], !DIExpression(DW_OP_consts, 1, DW_OP_plus, DW_OP_stack_value),
; CHECK: ![[i]] = !DILocalVariable(name: "i"
; CHECK: ![[comp]] = !DILocalVariable(name: "comp"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.data = private unnamed_addr constant [16 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15], align 16

define dso_local void @_Z21mul_pow_of_2_to_shiftjPjj(i32 %size, ptr nocapture %data, i32 %multiplicand) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %size, metadata !12, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata ptr %data, metadata !14, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata i32 %multiplicand, metadata !15, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !13
  br label %while.cond, !dbg !13

while.cond:                                       ; preds = %while.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.body ], !dbg !13
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !16, metadata !DIExpression()), !dbg !13
  %cmp = icmp ult i32 %i.0, %size, !dbg !13
  br i1 %cmp, label %while.body, label %while.end, !dbg !13

while.body:                                       ; preds = %while.cond
  %mul = mul i32 %i.0, %multiplicand, !dbg !17
  call void @llvm.dbg.value(metadata i32 %mul, metadata !19, metadata !DIExpression()), !dbg !17
  %idxprom = zext i32 %i.0 to i64, !dbg !17
  %arrayidx = getelementptr inbounds i32, ptr %data, i64 %idxprom, !dbg !17
  store i32 %mul, ptr %arrayidx, align 4, !dbg !17
  %inc = add nuw i32 %i.0, 1, !dbg !17
  call void @llvm.dbg.value(metadata i32 %inc, metadata !16, metadata !DIExpression()), !dbg !13
  br label %while.cond, !dbg !13, !llvm.loop !20

while.end:                                        ; preds = %while.cond
  ret void, !dbg !13
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "basic.cpp", directory: "/test")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 13.0.0"}
!7 = distinct !DISubprogram(name: "mul_pow_of_2_to_shift", linkageName: "_Z21mul_pow_of_2_to_shiftjPjj", scope: !1, file: !1, line: 17, type: !8, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11, !10}
!10 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!12 = !DILocalVariable(name: "size", arg: 1, scope: !7, file: !1, line: 17, type: !10)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocalVariable(name: "data", arg: 2, scope: !7, file: !1, line: 17, type: !11)
!15 = !DILocalVariable(name: "multiplicand", arg: 3, scope: !7, file: !1, line: 17, type: !10)
!16 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 18, type: !10)
!17 = !DILocation(line: 21, column: 27, scope: !18)
!18 = distinct !DILexicalBlock(scope: !7, file: !1, line: 20, column: 22)
!19 = !DILocalVariable(name: "comp", scope: !18, file: !1, line: 21, type: !10)
!20 = distinct !{!20, !21, !22, !23, !24}
!21 = !DILocation(line: 20, column: 5, scope: !7)
!22 = !DILocation(line: 24, column: 5, scope: !7)
!23 = !{!"llvm.loop.mustprogress"}
!24 = !{!"llvm.loop.vectorize.width", i32 1}
