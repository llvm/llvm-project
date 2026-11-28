; RUN: llc -mtriple=aarch64-unknown-linux-gnu -global-isel -stop-after=instruction-select %s -o - | FileCheck %s

; The sign/zero extends feeding the multiply are folded into a single
; SMADDLrrr/UMADDLrrr (smull/umull). The extended value is described by a
; dbg_value; when the dead G_SEXT/G_ZEXT is erased during instruction
; selection, the variable location must be salvaged in terms of the 32-bit
; source register using a sign/zero extension DIExpression.

; CHECK-DAG: ![[WIDE_S:[0-9]+]] = !DILocalVariable(name: "wide_s"
; CHECK-DAG: ![[WIDE_U:[0-9]+]] = !DILocalVariable(name: "wide_u"

define i64 @smull_dbg(i32 %a, i32 %b) !dbg !5 {
; CHECK-LABEL: name: smull_dbg
; CHECK:       [[A:%[0-9]+]]:gpr32 = COPY $w0
; CHECK:       [[B:%[0-9]+]]:gpr32 = COPY $w1
; CHECK:       DBG_VALUE [[A]], $noreg, ![[WIDE_S]], !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_stack_value)
; CHECK-NEXT:  {{%[0-9]+}}:gpr64 = nsw SMADDLrrr [[A]], [[B]], $xzr
entry:
  %ext.a = sext i32 %a to i64, !dbg !10
    #dbg_value(i64 %ext.a, !9, !DIExpression(), !10)
  %ext.b = sext i32 %b to i64, !dbg !10
  %mul = mul nsw i64 %ext.a, %ext.b, !dbg !11
  ret i64 %mul, !dbg !12
}

define i64 @umull_dbg(i32 %a, i32 %b) !dbg !15 {
; CHECK-LABEL: name: umull_dbg
; CHECK:       [[A:%[0-9]+]]:gpr32 = COPY $w0
; CHECK:       [[B:%[0-9]+]]:gpr32 = COPY $w1
; CHECK:       DBG_VALUE [[A]], $noreg, ![[WIDE_U]], !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_stack_value)
; CHECK-NEXT:  {{%[0-9]+}}:gpr64 = nuw UMADDLrrr [[A]], [[B]], $xzr
entry:
  %ext.a = zext i32 %a to i64, !dbg !20
    #dbg_value(i64 %ext.a, !19, !DIExpression(), !20)
  %ext.b = zext i32 %b to i64, !dbg !20
  %mul = mul nuw i64 %ext.a, %ext.b, !dbg !21
  ret i64 %mul, !dbg !22
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "smull_dbg", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{!13, !4, !4}
!8 = !{!9}
!9 = !DILocalVariable(name: "wide_s", scope: !5, file: !1, line: 2, type: !13)
!10 = !DILocation(line: 2, column: 3, scope: !5)
!11 = !DILocation(line: 3, column: 3, scope: !5)
!12 = !DILocation(line: 4, column: 3, scope: !5)
!13 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!14 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!15 = distinct !DISubprogram(name: "umull_dbg", scope: !1, file: !1, line: 6, type: !16, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{!14, !23, !23}
!18 = !{!19}
!19 = !DILocalVariable(name: "wide_u", scope: !15, file: !1, line: 7, type: !14)
!20 = !DILocation(line: 7, column: 3, scope: !15)
!21 = !DILocation(line: 8, column: 3, scope: !15)
!22 = !DILocation(line: 9, column: 3, scope: !15)
!23 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
