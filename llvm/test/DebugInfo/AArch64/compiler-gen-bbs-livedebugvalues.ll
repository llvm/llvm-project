; RUN: llc -O0 -regalloc=fast -stop-after=livedebugvalues -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not=DBG_VALUE

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios12.1.0"

declare void @use(i32 %x)

define void @f1(i32 %x) !dbg !6 {
; CHECK-LABEL: name: f1
entry:
; CHECK-LABEL: bb.0.entry:
  %var = add i32 %x, 1, !dbg !12
  call void @llvm.dbg.value(metadata i32 %var, metadata !9, metadata !DIExpression()), !dbg !12
; CHECK: DBG_VALUE renamable $w0, $noreg, !9, !DIExpression(), debug-location !12
; CHECK-NEXT: STRWui killed $w0, $sp, 3 :: (store 4 into %stack.0)
; CHECK-NEXT: DBG_VALUE $sp, 0, !9, !DIExpression(DW_OP_plus_uconst, 12)

  br label %artificial-bb-1, !dbg !13

artificial-bb-1:                                  ; preds = %entry
; CHECK-LABEL: bb.1.artificial-bb-1:
; CHECK: DBG_VALUE $sp, 0, !9, !DIExpression(DW_OP_plus_uconst, 12)

  br label %artificial-bb-2

artificial-bb-2:                                  ; preds = %artificial-bb-1
; CHECK-LABEL: bb.2.artificial-bb-2:
; CHECK: DBG_VALUE $sp, 0, !9, !DIExpression(DW_OP_plus_uconst, 12)

  %invisible = add i32 %var, 1
  br label %return, !dbg !14

return:                                           ; preds = %artificial-bb-2
; CHECK-LABEL: bb.3.return:
; CHECK: DBG_VALUE $sp, 0, !9, !DIExpression(DW_OP_plus_uconst, 12)

  call void @use(i32 %var)
  ret void, !dbg !15
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "compiler-gen-bbs-livedebugvalues.ll", directory: "/")
!2 = !{}
!3 = !{i32 6}
!4 = !{i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "f1", linkageName: "f1", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9, !11}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !6, file: !1, line: 4, type: !10)
!12 = !DILocation(line: 1, column: 1, scope: !6)
!13 = !DILocation(line: 2, column: 1, scope: !6)
!14 = !DILocation(line: 0, column: 1, scope: !6)
!15 = !DILocation(line: 4, column: 1, scope: !6)
