; RUN: opt -S -loop-reduce %s | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -S -loop-reduce %s | FileCheck %s
; REQUIRES: x86-registered-target

;; Ensure that SCEV-based salvaging in Loop Strength Reduction can salvage
;; variadic dbg.value intrinsics. Generated from the following C++:

;; clang -S -emit-llvm -Xclang -disable-llvm-passes -g lsr-variadic.cpp -o
;; Then running 'opt -O2' up until LSR.
;; void mul_to_addition(unsigned k, unsigned l, unsigned m, unsigned size, unsigned *data) {
;;     unsigned i = 0;
;; #pragma clang loop vectorize(disable)
;;     while (i < size) {
;;         unsigned comp = (4 * i) + k;
;;         unsigned comp2 = comp * l;
;;         unsigned comp3 = comp2 << m;
;;         data[i] = comp;
;;         i++;
;;     }
;; }
;; This produces variadic dbg.value intrinsics with location op DIArglists 
;; of length two and three.
;; A fourth dbg.value was added artificially by copying a generated dbg.value
;; and the modifying the position of the optimised-out value in the location
;; list.

; CHECK: #dbg_value(!DIArgList(i64 %lsr.iv, i32 %k), ![[comp:[0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_consts, 4, DW_OP_div, DW_OP_consts, 4, DW_OP_mul, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value),
; CHECK: #dbg_value(!DIArgList(i64 %lsr.iv, i32 %l, i32 %k), ![[comp2:[0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_consts, 4, DW_OP_div, DW_OP_consts, 4, DW_OP_mul, DW_OP_LLVM_arg, 2, DW_OP_plus, DW_OP_LLVM_arg, 1, DW_OP_mul, DW_OP_stack_value),
; CHECK: #dbg_value(!DIArgList(i64 %lsr.iv, i32 %m, i32 %l, i32 %k), ![[comp3:[0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_consts, 4, DW_OP_div, DW_OP_consts, 4, DW_OP_mul, DW_OP_LLVM_arg, 3, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_mul, DW_OP_LLVM_arg, 1, DW_OP_shl, DW_OP_stack_value),
; CHECK: #dbg_value(!DIArgList(i64 %lsr.iv, i32 %m, i32 %l, i32 %k), ![[comp3:[0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_consts, 4, DW_OP_div, DW_OP_consts, 4, DW_OP_mul, DW_OP_LLVM_arg, 3, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_mul, DW_OP_LLVM_arg, 1, DW_OP_shl, DW_OP_stack_value),
; CHECK: ![[comp]] = !DILocalVariable(name: "comp"
; CHECK: ![[comp2]] = !DILocalVariable(name: "comp2"
; CHECK: ![[comp3]] = !DILocalVariable(name: "comp3"


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z15mul_to_additionjjjjPj(i32 %k, i32 %l, i32 %m, i32 %size, ptr nocapture %data) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %k, metadata !14, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 %l, metadata !15, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 %m, metadata !16, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 %size, metadata !17, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata ptr %data, metadata !18, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !24
  %cmp9.not = icmp eq i32 %size, 0, !dbg !25
  br i1 %cmp9.not, label %while.end, label %while.body.preheader, !dbg !26

while.body.preheader:                             ; preds = %entry
  %wide.trip.count = zext i32 %size to i64, !dbg !25
  br label %while.body, !dbg !26

while.body:                                       ; preds = %while.body, %while.body.preheader
  %indvars.iv = phi i64 [ 0, %while.body.preheader ], [ %indvars.iv.next, %while.body ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !19, metadata !DIExpression()), !dbg !24
  %0 = trunc i64 %indvars.iv to i32, !dbg !27
  %mul = shl i32 %0, 2, !dbg !27
  %add = add i32 %mul, %k, !dbg !28
  call void @llvm.dbg.value(metadata i32 %add, metadata !20, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.value(metadata !DIArgList(i32 %add, i32 %l), metadata !22, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_mul, DW_OP_stack_value)), !dbg !29
  call void @llvm.dbg.value(metadata !DIArgList(i32 %add, i32 %m, i32 %l), metadata !23, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 2, DW_OP_mul, DW_OP_LLVM_arg, 1, DW_OP_shl, DW_OP_stack_value)), !dbg !29
  call void @llvm.dbg.value(metadata !DIArgList(i32 %m, i32 %add, i32 %l), metadata !23, metadata !DIExpression(DW_OP_LLVM_arg, 1, DW_OP_LLVM_arg, 2, DW_OP_mul, DW_OP_LLVM_arg, 0, DW_OP_shl, DW_OP_stack_value)), !dbg !29
  %arrayidx = getelementptr inbounds i32, ptr %data, i64 %indvars.iv, !dbg !30
  store i32 %add, ptr %arrayidx, align 4, !dbg !31
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !32
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !19, metadata !DIExpression()), !dbg !24
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count, !dbg !25
  br i1 %exitcond, label %while.body, label %while.end.loopexit, !dbg !26, !llvm.loop !33

while.end.loopexit:                               ; preds = %while.body
  br label %while.end, !dbg !37

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void, !dbg !37
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { "target-cpu"="x86-64" }


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "variadic2.cpp", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "mul_to_addition", linkageName: "_Z15mul_to_additionjjjjPj", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!8 = !DIFile(filename: "./variadic2.cpp", directory: "/test")
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !11, !11, !11, !12}
!11 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!13 = !{!14, !15, !16, !17, !18, !19, !20, !22, !23}
!14 = !DILocalVariable(name: "k", arg: 1, scope: !7, file: !8, line: 1, type: !11)
!15 = !DILocalVariable(name: "l", arg: 2, scope: !7, file: !8, line: 1, type: !11)
!16 = !DILocalVariable(name: "m", arg: 3, scope: !7, file: !8, line: 1, type: !11)
!17 = !DILocalVariable(name: "size", arg: 4, scope: !7, file: !8, line: 1, type: !11)
!18 = !DILocalVariable(name: "data", arg: 5, scope: !7, file: !8, line: 1, type: !12)
!19 = !DILocalVariable(name: "i", scope: !7, file: !8, line: 2, type: !11)
!20 = !DILocalVariable(name: "comp", scope: !21, file: !8, line: 5, type: !11)
!21 = distinct !DILexicalBlock(scope: !7, file: !8, line: 4, column: 23)
!22 = !DILocalVariable(name: "comp2", scope: !21, file: !8, line: 6, type: !11)
!23 = !DILocalVariable(name: "comp3", scope: !21, file: !8, line: 7, type: !11)
!24 = !DILocation(line: 0, scope: !7)
!25 = !DILocation(line: 4, column: 15, scope: !7)
!26 = !DILocation(line: 4, column: 6, scope: !7)
!27 = !DILocation(line: 5, column: 29, scope: !21)
!28 = !DILocation(line: 5, column: 34, scope: !21)
!29 = !DILocation(line: 0, scope: !21)
!30 = !DILocation(line: 8, column: 10, scope: !21)
!31 = !DILocation(line: 8, column: 18, scope: !21)
!32 = !DILocation(line: 9, column: 11, scope: !21)
!33 = distinct !{!33, !26, !34, !35, !36}
!34 = !DILocation(line: 10, column: 6, scope: !7)
!35 = !{!"llvm.loop.mustprogress"}
!36 = !{!"llvm.loop.vectorize.width", i32 1}
!37 = !DILocation(line: 11, column: 2, scope: !7)
