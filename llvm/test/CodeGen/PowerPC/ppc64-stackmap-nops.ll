; RUN: llc -verify-machineinstrs < %s -mcpu=ppc -mtriple=powerpc64-unknown-gnu-linux | FileCheck %s

define void @test_shadow_optimization() {
entry:
; Expect 12 bytes worth of nops here rather than 32: With the shadow optimization
; in place, 20 bytes will be consumed by the frame teardown and return instr.
; CHECK-LABEL: test_shadow_optimization:

; CHECK:      nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NOT:  nop
; CHECK: addi 1, 1, 64
; CHECK: ld [[REG1:[0-9]+]], 16(1)
; CHECK: ld 31, -8(1)
; CHECK: mtlr [[REG1]]
; CHECK: blr

  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64  0, i32  32)
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)

define void @test_shadow_optimization_dbg_label() !dbg !4 {
entry:
; CHECK-LABEL: test_shadow_optimization_dbg_label:
; CHECK:      nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NOT:  nop
; CHECK:      blr
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 32)
  #dbg_label(!6, !7)
  ret void
}

define void @test_shadow_optimization_dbg_value() !dbg !8 {
entry:
; CHECK-LABEL: test_shadow_optimization_dbg_value:
; CHECK:      nop
; CHECK-NEXT: nop
; CHECK-NEXT: nop
; CHECK-NOT:  nop
; CHECK:      blr
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 32)
  #dbg_value(i32 0, !9, !DIExpression(), !10)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "stackmap-shadow.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test_shadow_optimization_dbg_label", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !2)
!6 = !DILabel(scope: !4, name: "after_stackmap", file: !1, line: 2)
!7 = !DILocation(line: 2, column: 1, scope: !4)
!8 = distinct !DISubprogram(name: "test_shadow_optimization_dbg_value", scope: !1, file: !1, line: 4, type: !5, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!9 = !DILocalVariable(name: "value", scope: !8, file: !1, line: 5, type: !11)
!10 = !DILocation(line: 5, column: 1, scope: !8)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
