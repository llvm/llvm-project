; RUN: llc --filetype=obj %s -o %t.dxbc
; RUN: llvm-objcopy --dump-section=ILDB=%t.bc %t.dxbc
; RUN: dxil-dis %t.bc -o - | FileCheck %s

target triple = "dxil-pc-shadermodel6.3-library"

; The label gets removed, so just check that the function is still emitted,
; and other debug info is still present.
; CHECK: define void @foo(i32 %i) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @llvm.dbg.value

define void @foo(i32 %i) !dbg !4 {
entry:
    #dbg_value(i32 %i, !9, !DIExpression(), !11)
  br label %label

label:
    #dbg_label(!8, !12)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "label.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8}
!8 = !DILabel(scope: !4, name: "label", file: !1, line: 2, column: 1)
!9 = !DILocalVariable(name: "i", arg: 1, scope: !4, file: !1, line: 4, type: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 1, column: 1, scope: !4)
!12 = !DILocation(line: 2, column: 1, scope: !4)
