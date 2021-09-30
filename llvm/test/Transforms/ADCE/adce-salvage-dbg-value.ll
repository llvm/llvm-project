;; Check that adce salvages debug info properly.
; RUN: opt -passes=adce -S < %s | FileCheck %s

; ModuleID = 'test.ll'
source_filename = "test.ll"

; Function Attrs: nounwind readnone
declare void @may_not_return(i32) #0

; Function Attrs: nounwind readnone willreturn
declare void @will_return(i32) #1

define void @test(i32 %a) !dbg !6 {
; CHECK-LABEL: @test(
; CHECK-NEXT:    [[B:%.*]] = add i32 [[A:%.*]], 1
; CHECK-NEXT:    call void @llvm.dbg.value(metadata i32 [[B]]
; CHECK-NEXT:    call void @may_not_return(i32 [[B]])
; CHECK-NEXT:    call void @llvm.dbg.value(metadata i32 [[B]], {{.*}}DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)
; CHECK-NEXT:    ret void
;
  %b = add i32 %a, 1, !dbg !12
  call void @llvm.dbg.value(metadata i32 %b, metadata !9, metadata !DIExpression()), !dbg !12
  call void @may_not_return(i32 %b), !dbg !13
  %c = add i32 %b, 1, !dbg !14
  call void @llvm.dbg.value(metadata i32 %c, metadata !11, metadata !DIExpression()), !dbg !14
  call void @will_return(i32 %c), !dbg !15
  ret void, !dbg !16
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone willreturn }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{}
!3 = !{i32 5}
!4 = !{i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test", linkageName: "test", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9, !11}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !6, file: !1, line: 3, type: !10)
!12 = !DILocation(line: 1, column: 1, scope: !6)
!13 = !DILocation(line: 2, column: 1, scope: !6)
!14 = !DILocation(line: 3, column: 1, scope: !6)
!15 = !DILocation(line: 4, column: 1, scope: !6)
!16 = !DILocation(line: 5, column: 1, scope: !6)
