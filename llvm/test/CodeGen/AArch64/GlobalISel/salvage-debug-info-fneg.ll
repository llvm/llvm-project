; RUN: llc -mtriple=aarch64 -global-isel -global-isel-abort=1 -stop-after=instruction-select %s -o - | FileCheck %s

; Check that debug users of a G_FNEG folded into FNMUL are salvaged by flipping
; the sign bit of the source instead of being set to undef.

; CHECK-LABEL: name: fneg_f32
; CHECK: [[IN:%[0-9]+]]:fpr32 = LDRSui
; CHECK: DBG_VALUE [[IN]], $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_constu, 2147483648, DW_OP_xor, DW_OP_stack_value)
; CHECK: FNMULSrr [[IN]]
define void @fneg_f32(ptr %in, ptr %factor, ptr %out) !dbg !4 {
  %a = load volatile float, ptr %in, align 4, !dbg !11
  %b = load volatile float, ptr %factor, align 4, !dbg !11
  %neg = fneg float %a, !dbg !11
    #dbg_value(float %neg, !8, !DIExpression(), !11)
  %mul = fmul float %neg, %b, !dbg !11
  store volatile float %mul, ptr %out, align 4, !dbg !11
  ret void, !dbg !11
}

; CHECK-LABEL: name: fneg_f64
; CHECK: [[IN:%[0-9]+]]:fpr64 = LDRDui
; CHECK: DBG_VALUE [[IN]], $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_constu, 9223372036854775808, DW_OP_xor, DW_OP_stack_value)
; CHECK: FNMULDrr [[IN]]
define void @fneg_f64(ptr %in, ptr %factor, ptr %out) !dbg !12 {
  %a = load volatile double, ptr %in, align 8, !dbg !15
  %b = load volatile double, ptr %factor, align 8, !dbg !15
  %neg = fneg double %a, !dbg !15
    #dbg_value(double %neg, !13, !DIExpression(), !15)
  %mul = fmul double %neg, %b, !dbg !15
  store volatile double %mul, ptr %out, align 8, !dbg !15
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "fneg_f32", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!8 = !DILocalVariable(name: "neg", scope: !4, file: !1, line: 2, type: !7)
!11 = !DILocation(line: 2, column: 1, scope: !4)
!12 = distinct !DISubprogram(name: "fneg_f64", scope: !1, file: !1, line: 5, type: !5, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!13 = !DILocalVariable(name: "neg", scope: !12, file: !1, line: 6, type: !14)
!14 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!15 = !DILocation(line: 6, column: 1, scope: !12)
