; REQUIRES: asserts
; RUN: opt < %s -passes=globalopt -stats -disable-output 2>&1 | FileCheck %s
; CHECK: 1 globalopt - Number of global vars shrunk to booleans

source_filename = "test/Transforms/GlobalOpt/2009-03-05-dbg.ll"

@Stop = internal global i32 0, !dbg !0

; Function Attrs: nounwind ssp
define i32 @foo(i32 %i) #0 {
entry:
  %"alloca point" = bitcast i32 0 to i32
  call void @llvm.dbg.value(metadata i32 %i, metadata !8, metadata !12), !dbg !13
  %0 = icmp eq i32 %i, 1, !dbg !13
  br i1 %0, label %bb, label %bb1, !dbg !13

bb:                                               ; preds = %entry
  store i32 0, ptr @Stop, align 4, !dbg !15
  %1 = mul nsw i32 %i, 42, !dbg !16
  call void @llvm.dbg.value(metadata i32 %1, metadata !8, metadata !12), !dbg !16
  br label %bb2, !dbg !16

bb1:                                              ; preds = %entry
  store i32 1, ptr @Stop, align 4, !dbg !17
  br label %bb2, !dbg !17

bb2:                                              ; preds = %bb1, %bb
  %i_addr.0 = phi i32 [ %1, %bb ], [ %i, %bb1 ]
  br label %return, !dbg !18

return:                                           ; preds = %bb2
  ret i32 %i_addr.0, !dbg !18
}

; Function Attrs: nounwind readnone

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp
define i32 @bar() #0 {
entry:
  %"alloca point" = bitcast i32 0 to i32
  %0 = load i32, ptr @Stop, align 4, !dbg !19
  %1 = icmp eq i32 %0, 1, !dbg !19
  br i1 %1, label %bb, label %bb1, !dbg !19

bb:                                               ; preds = %entry

  br label %bb2, !dbg !24

bb1:                                              ; preds = %entry
  br label %bb2, !dbg !25

bb2:                                              ; preds = %bb1, %bb
  %.0 = phi i32 [ 0, %bb ], [ 1, %bb1 ]
  br label %return, !dbg !25

return:                                           ; preds = %bb2
  ret i32 %.0, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "Stop", scope: !2, file: !3, line: 2, type: !5, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C89, file: !3, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "g.c", directory: "/tmp")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 2}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !DILocalVariable(name: "i", arg: 1, scope: !9, file: !3, line: 4, type: !5)
!9 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !2, file: !3, line: 4, type: !10, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !2)!10 = !DISubroutineType(types: !11)!11 = !{!5, !5}!12 = !DIExpression()!13 = !DILocation(line: 5, scope: !14)!14 = distinct !DILexicalBlock(scope: !9, file: !3)!15 = !DILocation(line: 6, scope: !14)!16 = !DILocation(line: 7, scope: !14)!17 = !DILocation(line: 9, scope: !14)!18 = !DILocation(line: 11, scope: !14)!19 = !DILocation(line: 14, scope: !20)!20 = distinct !DILexicalBlock(scope: !21, file: !3)!21 = distinct !DISubprogram(name: "bar", linkageName: "bar", scope: !2, file: !3, line: 13, type: !22, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !2)!22 = !DISubroutineType(types: !23)!23 = !{!5}!24 = !DILocation(line: 15, scope: !20)!25 = !DILocation(line: 16, scope: !20)
