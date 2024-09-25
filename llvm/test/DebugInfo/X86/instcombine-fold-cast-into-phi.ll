; RUN: opt -passes=instcombine -S %s | FileCheck %s

;; Instcombine folds the trunc %x into the phi. Check it updates the phi's dbg
;; user (otherwise the dbg use becomes poison after the original phi is
;; deleted). Check the new phi inherits the DebugLoc.

; CHECK: %[[phi:.*]] = phi i8 [ 1, %{{.*}} ], [ 0, %{{.*}} ], !dbg ![[dbg:[0-9]+]]
; CHECK: #dbg_value(i8 %[[phi]], ![[#]], !DIExpression(DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_stack_value)
; CHECK: ![[dbg]] = !DILocation(line: 123,

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define linkonce_odr float @f(i1 %cond) {
entry:
  br i1 %cond, label %if.then, label %if.end

if.then:                                          ; preds = entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %p.0 = phi i32 [ 1, %if.then ], [ 0, %entry ], !dbg !13
  call void @llvm.dbg.value(metadata i32 %p.0, metadata !4, metadata !DIExpression()), !dbg !13
  %x = trunc i32 %p.0 to i8
  %callff = call float @ff(i8  %x)
  ret float %callff
}

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare float @ff(float)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_11, file: !1, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "btConvexTriangleMeshShape.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "p", scope: !5, file: !6, line: 2057, type: !12)
!5 = distinct !DILexicalBlock(scope: !7, file: !6, line: 2056, column: 4)
!6 = !DIFile(filename: "reduce.cpp", directory: "/")
!7 = distinct !DISubprogram(name: "diagonalize", linkageName: "_ZN11btMatrix3x311diagonalizeERS_fi", scope: !8, file: !6, line: 2054, type: !9, scopeLine: 2055, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !11, retainedNodes: !2)
!8 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "btMatrix3x3", file: !6, line: 2050, size: 24, flags: DIFlagTypePassByValue, elements: !2, identifier: "_ZTS11btMatrix3x3")
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DISubprogram(name: "diagonalize", linkageName: "_ZN11btMatrix3x311diagonalizeERS_fi", scope: !8, file: !6, line: 2054, type: !9, scopeLine: 2054, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 123, scope: !5)
