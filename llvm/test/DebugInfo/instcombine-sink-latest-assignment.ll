; RUN: opt %s -o - -S --passes=instcombine | FileCheck %s
; RUN: opt %s -o - -S --passes=instcombine --try-experimental-debuginfo-iterators | FileCheck %s
;
; CHECK-LABEL: for.body:
; CHECK-NEXT:  %sub.ptr.rhs.cast.i.i = ptrtoint ptr %call2.i.i to i64,
; CHECK-NEXT:  #dbg_value(i64 %sub.ptr.rhs.cast.i.i, !{{[0-9]*}}, !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value)
;
;; The code below is representative of a common situation: where we've had a
;; loop be completely optimised out, leaving dbg.values representing the
;; assignments for it, including a rotated assignment to the loop counter.
;; Below, it's the two dbg.values in the entry block, assigning first the
;; value of %conv.i, then the value of %conv.i minus one.
;;
;; When instcombine sinks %conv.i, it's critical that if it sinks a dbg.value
;; with it, it sinks the most recent assignment. Otherwise it will re-order the
;; assignments below, and a fall counter assignment will continue on from the
;; end of the optimised-out loop.
;;
;; The check lines test that when the trunc sinks (along with the ptrtoint),
;; we get the dbg.value with a DW_OP_minus in it.

; ModuleID = 'tmp.ll'
source_filename = "tmp.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

define void @_ZN4llvm16MCObjectStreamer15EmitInstructionERKNS_6MCInstE(i1 %tobool.not) local_unnamed_addr {
entry:
  %call2.i.i = load volatile ptr, ptr null, align 8, !dbg !4
  %sub.ptr.rhs.cast.i.i = ptrtoint ptr %call2.i.i to i64, !dbg !4
  %conv.i = trunc i64 %sub.ptr.rhs.cast.i.i to i32, !dbg !4
  tail call void @llvm.dbg.value(metadata i32 %conv.i, metadata !16, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %conv.i, metadata !16, metadata !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !18
  br i1 %tobool.not, label %common.ret, label %for.body

common.ret:                                       ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry
  %call2 = call ptr @_ZNK4llvm6MCInst10getOperandEj(i32 %conv.i)
  br label %common.ret
}

declare i32 @_ZNK4llvm6MCInst14getNumOperandsEv()

declare ptr @_ZNK4llvm6MCInst10getOperandEj(i32) local_unnamed_addr

declare i64 @_ZNK4llvm25SmallVectorTemplateCommonINS_9MCOperandEvE4sizeEv()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.cpp", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocation(line: 197, column: 26, scope: !5)
!5 = distinct !DILexicalBlock(scope: !7, file: !6, line: 197, column: 3)
!6 = !DIFile(filename: "foo.cpp", directory: ".")
!7 = distinct !DISubprogram(name: "EmitInstruction", linkageName: "_ZN4llvm16MCObjectStreamer15EmitInstructionERKNS_6MCInstE", scope: !8, file: !6, line: 195, type: !13, scopeLine: 195, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !15, retainedNodes: !2)
!8 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "MCObjectStreamer", scope: !10, file: !9, line: 33, size: 2432, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !2, vtableHolder: !11)
!9 = !DIFile(filename: "bar.h", directory: ".")
!10 = !DINamespace(name: "llvm", scope: null)
!11 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "MCStreamer", scope: !10, file: !12, line: 108, size: 2240, flags: DIFlagFwdDecl | DIFlagNonTrivial, identifier: "_ZTSN4llvm10MCStreamerE")
!12 = !DIFile(filename: "baz.h", directory: ".")
!13 = distinct !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DISubprogram(name: "EmitInstruction", linkageName: "_ZN4llvm16MCObjectStreamer15EmitInstructionERKNS_6MCInstE", scope: !8, file: !9, line: 88, type: !13, scopeLine: 88, containingType: !8, virtualIndex: 86, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagVirtual | DISPFlagOptimized)
!16 = !DILocalVariable(name: "i", scope: !5, file: !6, line: 197, type: !17)
!17 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!18 = !DILocation(line: 0, scope: !5)
