; RUN: llc -start-after=codegenprepare -stop-before=finalize-isel < %s -o - | FileCheck %s

; Test that stack frame dbg.values are lowered to DBG_VALUEs, in blocks that
; are local to the alloca, and elsewhere. Differs from dbg-value-frame-index.ll
; because this test does not result in the frame-index being in a vreg,
; instead it's exclusively referred to by memory operands of instructions.
;
; Additionally test that we don't re-order with constant values -- both are
; independent of the order the instructions get lowered, but should not
; interleave.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

declare void @dud()

; CHECK: [[BARVAR:![0-9]+]] = !DILocalVariable(name: "bar",

define i32 @foo() !dbg !6 {
; CHECK-LABEL: body

; CHECK:      DBG_VALUE 0, $noreg, [[BARVAR]]
; CHECK-NEXT: MOV32mi %[[STACKLOC:[a-zA-Z0-9\.]+]], 1, $noreg
; CHECK-NEXT: DBG_VALUE %[[STACKLOC]], $noreg, [[BARVAR]]

  %p1 = alloca i32
  call void @llvm.dbg.value(metadata i32 *null, metadata !17, metadata !DIExpression()), !dbg !18
  store i32 0, i32 *%p1
  call void @llvm.dbg.value(metadata i32 *%p1, metadata !17, metadata !DIExpression()), !dbg !18
  br label %foo

foo:

; CHECK-LABEL: bb.1.foo
; CHECK:      DBG_VALUE %[[STACKLOC]], $noreg, [[BARVAR]]

  call void @dud()
  call void @llvm.dbg.value(metadata i32 *%p1, metadata !17, metadata !DIExpression()), !dbg !18
  br label %bar

bar:

; CHECK-LABEL: bb.2.bar
; CHECK:      DBG_VALUE %[[STACKLOC]], $noreg, [[BARVAR]]
; CHECK-NEXT: ADJCALLSTACKDOWN
; CHECK-NEXT: CALL
; CHECK-NEXT: ADJCALLSTACKUP
; CHECK-NEXT: DBG_VALUE 0, $noreg, [[BARVAR]]
  call void @llvm.dbg.value(metadata i32 *%p1, metadata !17, metadata !DIExpression()), !dbg !18
  call void @dud()
  call void @llvm.dbg.value(metadata i32 *null, metadata !17, metadata !DIExpression()), !dbg !18
  %loaded = load i32, i32 *%p1
  ret i32 %loaded, !dbg !19
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #6

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2)
!1 = !DIFile(filename: "a.c", directory: "b")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 13, type: !7, isLocal: false, isDefinition: true, scopeLine: 14, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!140}
!140 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DISubrange(count: 5)
!17 = !DILocalVariable(name: "bar", scope: !6, line: 13, type: !140)
!18 = !DILocation(line: 13, column: 23, scope: !6)
!19 = !DILocation(line: 15, column: 5, scope: !6)
!20 = !DILocation(line: 16, column: 1, scope: !6)
!21 = !DILocalVariable(name: "baz", scope: !6, line: 13, type: !140)
