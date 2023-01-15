; RUN: opt -S -loop-reduce %s -o - | FileCheck %s
; REQUIRES: x86-registered-target

;; Ensure that we retain debuginfo for the induction variable and dependant
;; variables when loop strength reduction is applied to the loop.
;; This IR produced from:
;;
;; clang -S -emit-llvm -Xclang -disable-llvm-passes -g lsr-basic.cpp -o
;; Then executing opt -O2 up to the the loopFullUnroll pass.
;; void mul_pow_of_2_to_shift(unsigned size, unsigned *data) {
;; unsigned i = 0;
;; #pragma clang loop vectorize(disable)
;;    while (i < size) {
;;         unsigned comp = i * 8;
;;         data[i] = comp;
;;        i++;                // DexLabel('mul_pow_of_2_induction_increment')
;;   }
;; }
; CHECK: call void @llvm.dbg.value(metadata i64 %lsr.iv, metadata ![[i:[0-9]+]], metadata !DIExpression(DW_OP_consts, 8, DW_OP_div, DW_OP_stack_value))
; CHECK: call void @llvm.dbg.value(metadata i64 %lsr.iv, metadata ![[comp:[0-9]+]], metadata !DIExpression(DW_OP_consts, 8, DW_OP_div, DW_OP_consts, 8, DW_OP_mul, DW_OP_stack_value))
; CHECK: call void @llvm.dbg.value(metadata i64 %lsr.iv, metadata ![[i]], metadata !DIExpression(DW_OP_consts, 8, DW_OP_div, DW_OP_consts, 1, DW_OP_plus, DW_OP_stack_value))
; CHECK: ![[i]] = !DILocalVariable(name: "i"
; CHECK: ![[comp]] = !DILocalVariable(name: "comp"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z21mul_pow_of_2_to_shiftjPj(i32 %size, ptr nocapture %data) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %size, metadata !12, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata ptr %data, metadata !14, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !13
  %cmp4.not = icmp eq i32 %size, 0, !dbg !13
  br i1 %cmp4.not, label %while.end, label %while.body.preheader, !dbg !13

while.body.preheader:                             ; preds = %entry
  %wide.trip.count = zext i32 %size to i64, !dbg !13
  br label %while.body, !dbg !13

while.body:                                       ; preds = %while.body, %while.body.preheader
  %indvars.iv = phi i64 [ 0, %while.body.preheader ], [ %indvars.iv.next, %while.body ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !15, metadata !DIExpression()), !dbg !13
  %0 = trunc i64 %indvars.iv to i32, !dbg !16
  %mul = shl i32 %0, 3, !dbg !16
  call void @llvm.dbg.value(metadata i32 %mul, metadata !18, metadata !DIExpression()), !dbg !16
  %arrayidx = getelementptr inbounds i32, ptr %data, i64 %indvars.iv, !dbg !16
  store i32 %mul, ptr %arrayidx, align 4, !dbg !16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !16
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !15, metadata !DIExpression()), !dbg !13
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count, !dbg !13
  br i1 %exitcond, label %while.body, label %while.end.loopexit, !dbg !13, !llvm.loop !19

while.end.loopexit:                               ; preds = %while.body
  br label %while.end, !dbg !13

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void, !dbg !13
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "lsr-basic.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 13.0.0"}
!7 = distinct !DISubprogram(name: "mul_pow_of_2_to_shift", linkageName: "_Z21mul_pow_of_2_to_shiftjPj", scope: !1, file: !1, line: 18, type: !8, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11}
!10 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!12 = !DILocalVariable(name: "size", arg: 1, scope: !7, file: !1, line: 18, type: !10)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocalVariable(name: "data", arg: 2, scope: !7, file: !1, line: 18, type: !11)
!15 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 19, type: !10)
!16 = !DILocation(line: 22, column: 27, scope: !17)
!17 = distinct !DILexicalBlock(scope: !7, file: !1, line: 21, column: 22)
!18 = !DILocalVariable(name: "comp", scope: !17, file: !1, line: 22, type: !10)
!19 = distinct !{!19, !20, !21, !22, !23}
!20 = !DILocation(line: 21, column: 5, scope: !7)
!21 = !DILocation(line: 25, column: 5, scope: !7)
!22 = !{!"llvm.loop.mustprogress"}
!23 = !{!"llvm.loop.vectorize.width", i32 1}
