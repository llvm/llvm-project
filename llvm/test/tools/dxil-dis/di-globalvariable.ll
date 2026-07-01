; RUN: llc --filetype=obj %s -o %t.dxbc
; RUN: llvm-objcopy --dump-section=ILDB=%t.bc %t.dxbc
; RUN: dxil-dis %t.bc -o - | FileCheck %s

target triple = "dxil-unknown-shadermodel6.3-library"

;; @x is optimized away. DIGLobalVariable stays, but its variable operand is null (not printed).
;; @y is not optimized away.

@x = global i32 0, align 4, !dbg !0
@y = global i32 1, align 4, !dbg !2

define void @foo() {
  %y = load i32, ptr @y
  ret void
}

; CHECK-DAG: !llvm.dbg.cu = !{![[CU:[0-9]+]]}
; CHECK-DAG: ![[CU]] = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: 1, globals: ![[GLOBALS:[0-9]+]])
; CHECK-DAG: ![[GLOBALS]] = !{![[GVX:[0-9]+]], ![[GVY:[0-9]+]]}
; CHECK-DAG: ![[GVX]] = !DIGlobalVariable(name: "x", scope: ![[CU]], file: ![[FILE:[0-9]+]], line: 1, type: ![[TYPE:[0-9]+]], isLocal: false, isDefinition: true)
; CHECK-DAG: ![[GVY]] = !DIGlobalVariable(name: "y", scope: ![[CU]], file: ![[FILE:[0-9]+]], line: 1, type: ![[TYPE:[0-9]+]], isLocal: false, isDefinition: true, variable: i32* @y)
; CHECK-DAG: ![[FILE]] = !DIFile(filename: "cu.cpp", directory: "/tmp")
; CHECK-DAG: ![[TYPE]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!8, !9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: !4, file: !5, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = !DIGlobalVariable(name: "y", scope: !4, file: !5, line: 1, type: !7, isLocal: false, isDefinition: true)
!4 = distinct !DICompileUnit(language: DW_LANG_C, file: !5, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !6, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!5 = !DIFile(filename: "cu.cpp", directory: "/tmp")
!6 = !{!0, !2}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
