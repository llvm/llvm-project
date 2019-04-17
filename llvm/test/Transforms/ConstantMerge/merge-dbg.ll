; RUN: opt < %s -constmerge -S | FileCheck %s

; CHECK: = constant i32 1, !dbg [[A:![0-9]+]], !dbg [[B:![0-9]+]]
@a = internal constant i32 1, !dbg !0
@b = unnamed_addr constant i32 1, !dbg !9

define void @test1(i32** %P1, i32** %P2) {
  store i32* @a, i32** %P1
  store i32* @b, i32** %P2
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}

; CHECK: [[A]] = !DIGlobalVariableExpression(var: [[VA:![0-9]+]], expr: !DIExpression())
; CHECK: [[VA]] = distinct !DIGlobalVariable(name: "y"
; CHECK: [[B]] = !DIGlobalVariableExpression(var: [[VB:![0-9]+]], expr: !DIExpression())
; CHECK: [[VB]] = distinct !DIGlobalVariable(name: "x"

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 5.0.0 (trunk 297227) (llvm/trunk 297234)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "1.cc", directory: "/build")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}

!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
