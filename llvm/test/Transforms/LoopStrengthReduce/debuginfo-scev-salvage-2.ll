; RUN: opt -S -loop-reduce %s -o - | FileCheck %s
; REQUIRES: x86-registered-target

;; Ensure that we retain debuginfo for the induction variable and dependant
;; variables when loop strength reduction is applied to the loop.
;; This IR produced from:
;;
;; clang -S -emit-llvm -Xclang -disable-llvm-passes -g lsr-basic.cpp -o
;; Then executing opt -O2 up to the the loopFullUnroll pass.
;; void basic_recurrence(unsigned k, unsigned size, unsigned *data)
;; {
;;    unsigned i = 0;
;; #pragma clang loop vectorize(disable)
;;    while (i < size) {
;;         unsigned comp = i * k; 
;;         data[i] = comp;
;;         i++;
;;    }
;; }
; CHECK: call void @llvm.dbg.value(metadata i64 %lsr.iv, metadata ![[i:[0-9]+]], metadata !DIExpression())
; CHECK: call void @llvm.dbg.value(metadata !DIArgList(i64 %lsr.iv, i32 %k), metadata ![[comp:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_mul, DW_OP_stack_value))
; CHECK: call void @llvm.dbg.value(metadata i64 %lsr.iv, metadata ![[i]],  metadata !DIExpression(DW_OP_consts, 1, DW_OP_plus, DW_OP_stack_value))
; CHECK: ![[i]] = !DILocalVariable(name: "i"
; CHECK: ![[comp]] = !DILocalVariable(name: "comp"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z16basic_recurrencejjPj(i32 %k, i32 %size, ptr nocapture %data) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %k, metadata !13, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 %size, metadata !15, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata ptr %data, metadata !16, metadata !DIExpression()), !dbg !14
  br label %while.cond, !dbg !14

while.cond:                                       ; preds = %while.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !17, metadata !DIExpression()), !dbg !14
  %cmp = icmp ult i32 %i.0, %size, !dbg !14
  br i1 %cmp, label %while.body, label %while.end, !dbg !14

while.body:                                       ; preds = %while.cond
  %mul = mul i32 %i.0, %k, !dbg !18
  call void @llvm.dbg.value(metadata i32 %mul, metadata !20, metadata !DIExpression()), !dbg !18
  %idxprom = zext i32 %i.0 to i64, !dbg !18
  %arrayidx = getelementptr inbounds i32, ptr %data, i64 %idxprom, !dbg !18
  store i32 %mul, ptr %arrayidx, align 4, !dbg !18
  %inc = add nuw i32 %i.0, 1, !dbg !18
  call void @llvm.dbg.value(metadata i32 %inc, metadata !17, metadata !DIExpression()), !dbg !14
  br label %while.cond, !dbg !14, !llvm.loop !21

while.end:                                        ; preds = %while.cond
  ret void, !dbg !14
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "basic.cpp", directory: "/test")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0)"}
!7 = distinct !DISubprogram(name: "basic_recurrence", linkageName: "_Z16basic_recurrencejjPj", scope: !8, file: !8, line: 82, type: !9, scopeLine: 83, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "./basic.cpp", directory: "/test")
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !11, !12}
!11 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!13 = !DILocalVariable(name: "k", arg: 1, scope: !7, file: !8, line: 82, type: !11)
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocalVariable(name: "size", arg: 2, scope: !7, file: !8, line: 82, type: !11)
!16 = !DILocalVariable(name: "data", arg: 3, scope: !7, file: !8, line: 82, type: !12)
!17 = !DILocalVariable(name: "i", scope: !7, file: !8, line: 84, type: !11)
!18 = !DILocation(line: 87, column: 27, scope: !19)
!19 = distinct !DILexicalBlock(scope: !7, file: !8, line: 86, column: 22)
!20 = !DILocalVariable(name: "comp", scope: !19, file: !8, line: 87, type: !11)
!21 = distinct !{!21, !22, !23, !24, !25}
!22 = !DILocation(line: 86, column: 5, scope: !7)
!23 = !DILocation(line: 90, column: 5, scope: !7)
!24 = !{!"llvm.loop.mustprogress"}
!25 = !{!"llvm.loop.vectorize.width", i32 1}
