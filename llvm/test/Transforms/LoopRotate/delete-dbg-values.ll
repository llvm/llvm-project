; RUN: opt --passes=loop-rotate -o - -S %s | FileCheck %s --implicit-check-not=dbg.value
; RUN: opt --passes=loop-rotate -o - -S %s --try-experimental-debuginfo-iterators | FileCheck %s --implicit-check-not=dbg.value
;
;; Test some fine-grained behaviour of loop-rotate's de-duplication of
;; dbg.values. The intrinsic on the first branch should be seen and
;; prevent the rotation of the dbg.value for "sink" into the entry block.
;; However the other dbg.value, for "source", should not be seen, and we'll
;; get a duplicate.
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; CHECK-LABEL: define void @_ZNK4llvm5APInt4sextEj(ptr
; CHECK-LABEL: entry:
; CHECK:       #dbg_value(i32 0, ![[SRC:[0-9]+]],
; CHECK-NEXT:  load
; CHECK-NEXT:  #dbg_value(i32 0, ![[SINK:[0-9]+]],
; CHECK-NEXT:  #dbg_value(i32 0, ![[SRC]],
; CHECK-LABEL: for.body:
; CHECK:       #dbg_value(i32 0, ![[SINK]],
; CHECK-NEXT:  #dbg_value(i32 0, ![[SRC]],

declare void @llvm.dbg.value(metadata, metadata, metadata)

define void @_ZNK4llvm5APInt4sextEj(ptr %agg.result) !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, metadata !4, metadata !DIExpression()), !dbg !10
  %.pre = load i32, ptr %agg.result, align 8
  tail call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ 1, %for.body ]
  tail call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !10
  tail call void @llvm.dbg.value(metadata i32 0, metadata !4, metadata !DIExpression()), !dbg !10
  %cmp12.not = icmp eq i32 %i.0, %.pre, !dbg !10
  br i1 %cmp12.not, label %for.end, label %for.body

for.body:                                         ; preds = %for.cond
  store i64 0, ptr %agg.result, align 8
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo", directory: "bar")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "source", scope: !5, file: !6, line: 170, type: !8)
!5 = distinct !DISubprogram(name: "ConvertUTF16toUTF32", scope: !6, file: !6, line: 166, type: !7, scopeLine: 168, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DIFile(filename: "fooo", directory: ".")
!7 = !DISubroutineType(types: !2)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!10 = !DILocation(line: 0, scope: !5)
!11 = !DILocalVariable(name: "sink", scope: !5, file: !6, line: 170, type: !8)
