; RUN: llc -mtriple=riscv32 -mattr=+zbkb -global-isel \
; RUN:   -global-isel-abort=1 -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+zbkb -global-isel \
; RUN:   -global-isel-abort=1 -stop-after=riscv-prelegalizer-combiner < %s \
; RUN:   | FileCheck %s

; CHECK: DBG_VALUE 0, $noreg, !{{[0-9]+}}, !DIExpression()

declare i8 @llvm.bitreverse.i8(i8)

define i8 @bitreverse_i8_zero() !dbg !4 {
  %r = call i8 @llvm.bitreverse.i8(i8 0)
    #dbg_value(i8 %r, !8, !DIExpression(), !9)
  ret i8 %r
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "debug-dead-def.ll", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "bitreverse_i8_zero", scope: !1, file: !1, line: 1, type: !5, unit: !0, retainedNodes: !6, spFlags: DISPFlagDefinition)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!8 = !DILocalVariable(name: "r", scope: !4, file: !1, line: 1, type: !10)
!9 = !DILocation(line: 1, scope: !4)
!10 = !DIBasicType(name: "i8", size: 8, encoding: DW_ATE_unsigned)
