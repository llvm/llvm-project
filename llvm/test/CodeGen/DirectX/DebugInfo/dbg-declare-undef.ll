; RUN: llc %s -o - | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; CHECK-LABEL: define void @foo() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call void @llvm.dbg.declare(metadata ptr poison,
; CHECK:         ret void
; CHECK-NEXT:  }

define void @foo() {
entry:
  %0 = alloca i32, align 4
    #dbg_declare(ptr %0, !4, !DIExpression(), !10)
  %1 = load i32, ptr %0
  store i32 %1, ptr %0
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "undef.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocalVariable(name: "i", scope: !5, file: !1, line: 2, type: !9)
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 2, column: 7, scope: !5)
