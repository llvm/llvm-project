; RUN: llc -mtriple=hexagon -insert-dist-cutoff=20 -verify-machineinstrs \
; RUN:   < %s | FileCheck %s
; RUN: opt -strip-debug < %s | \
; RUN:   llc -mtriple=hexagon -insert-dist-cutoff=20 -verify-machineinstrs | \
; RUN:   FileCheck %s

define i32 @geninsert_ignores_debug_instrs_across_blocks(
    i1 %cond, i8 %a0, i8 %a1, i8 %a2, i8 %a3) #0 !dbg !3 {
; CHECK-LABEL: geninsert_ignores_debug_instrs_across_blocks:
; CHECK-COUNT-2: = insert(
; CHECK-NOT: = insert(
; CHECK: = combine(
entry:
  #dbg_value(i32 0, !25, !DIExpression(), !17)
  #dbg_value(i32 0, !26, !DIExpression(), !17)
  br i1 %cond, label %debug, label %plain

debug:
  #dbg_value(i32 0, !6, !DIExpression(), !17)
  #dbg_value(i32 0, !7, !DIExpression(), !17)
  #dbg_value(i32 0, !8, !DIExpression(), !17)
  #dbg_value(i32 0, !9, !DIExpression(), !17)
  #dbg_value(i32 0, !10, !DIExpression(), !17)
  #dbg_value(i32 0, !11, !DIExpression(), !17)
  #dbg_value(i32 0, !12, !DIExpression(), !17)
  #dbg_value(i32 0, !13, !DIExpression(), !17)
  #dbg_value(i32 0, !14, !DIExpression(), !17)
  #dbg_value(i32 0, !15, !DIExpression(), !17)
  call void @debug_side()
  br label %join

plain:
  call void @plain_side()
  br label %join

join:
  %v5 = insertelement <4 x i8> poison, i8 %a0, i32 0
  #dbg_value(i32 0, !22, !DIExpression(), !17)
  %v6 = insertelement <4 x i8> %v5, i8 %a1, i32 1
  #dbg_value(i32 0, !23, !DIExpression(), !17)
  %v7 = insertelement <4 x i8> %v6, i8 %a2, i32 2
  #dbg_value(i32 0, !24, !DIExpression(), !17)
  %v8 = insertelement <4 x i8> %v7, i8 %a3, i32 3
  %v9 = bitcast <4 x i8> %v8 to i32
  ret i32 %v9
}

declare void @debug_side()
declare void @plain_side()

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "geninsert-debug-distance.ll", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "geninsert_ignores_debug_instrs_across_blocks", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !5)
!4 = !DISubroutineType(types: !18)
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !22, !23, !24, !25, !26}
!6 = !DILocalVariable(name: "value0", scope: !3, file: !1, line: 1, type: !16)
!7 = !DILocalVariable(name: "value1", scope: !3, file: !1, line: 1, type: !16)
!8 = !DILocalVariable(name: "value2", scope: !3, file: !1, line: 1, type: !16)
!9 = !DILocalVariable(name: "value3", scope: !3, file: !1, line: 1, type: !16)
!10 = !DILocalVariable(name: "value4", scope: !3, file: !1, line: 1, type: !16)
!11 = !DILocalVariable(name: "value5", scope: !3, file: !1, line: 1, type: !16)
!12 = !DILocalVariable(name: "value6", scope: !3, file: !1, line: 1, type: !16)
!13 = !DILocalVariable(name: "value7", scope: !3, file: !1, line: 1, type: !16)
!14 = !DILocalVariable(name: "value8", scope: !3, file: !1, line: 1, type: !16)
!15 = !DILocalVariable(name: "value9", scope: !3, file: !1, line: 1, type: !16)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocation(line: 1, column: 1, scope: !3)
!18 = !{!16, !27, !19, !19, !19, !19}
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!22 = !DILocalVariable(name: "join0", scope: !3, file: !1, line: 1, type: !16)
!23 = !DILocalVariable(name: "join1", scope: !3, file: !1, line: 1, type: !16)
!24 = !DILocalVariable(name: "join2", scope: !3, file: !1, line: 1, type: !16)
!25 = !DILocalVariable(name: "entry0", scope: !3, file: !1, line: 1, type: !16)
!26 = !DILocalVariable(name: "entry1", scope: !3, file: !1, line: 1, type: !16)
!27 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
