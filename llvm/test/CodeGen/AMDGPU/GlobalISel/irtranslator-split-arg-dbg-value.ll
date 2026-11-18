; RUN: llc -global-isel -mtriple=amdgpu9.00-amd-amdhsa -stop-after=ir-translator < %s | FileCheck -check-prefix=IRT %s

; A dbg.value of an argument that the calling convention splits gets one
; fragment per part, as in SelectionDAG.

; The fragments are relative to the fragment the expression already has.
; IRT-LABEL: name: in_fragment
; IRT:      [[LO:%[0-9]+]]:_(i32) = COPY $vgpr0
; IRT-NEXT: [[HI:%[0-9]+]]:_(i32) = COPY $vgpr1
; IRT:      DBG_VALUE [[LO]](i32), $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 64, 32)
; IRT-NEXT: DBG_VALUE [[HI]](i32), $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 96, 32)
define hidden i64 @in_fragment(i64 %v) !dbg !10 {
    #dbg_value(i64 %v, !11, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !12)
  ret i64 %v, !dbg !12
}

; Bits past the end of the expression's fragment are dropped.
; IRT-LABEL: name: clipped
; IRT:      [[LO:%[0-9]+]]:_(i32) = COPY $vgpr0
; IRT-NEXT: [[HI:%[0-9]+]]:_(i32) = COPY $vgpr1
; IRT:      DBG_VALUE [[LO]](i32), $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; IRT-NEXT: DBG_VALUE [[HI]](i32), $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 32, 16)
define hidden i64 @clipped(i64 %v) !dbg !20 {
    #dbg_value(i64 %v, !21, !DIExpression(DW_OP_LLVM_fragment, 0, 48), !22)
  ret i64 %v, !dbg !22
}

; A part passed on the stack is described by its slot.
; IRT-LABEL: name: reg_and_stack
; IRT:      [[LO:%[0-9]+]]:_(i32) = COPY $vgpr30
; IRT:      DBG_VALUE [[LO]](i32), $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; IRT-NEXT: DBG_VALUE %fixed-stack.0, 0, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 32, 32)
define hidden i64 @reg_and_stack(<30 x i32> %pad, i64 %v) !dbg !30 {
    #dbg_value(i64 %v, !31, !DIExpression(), !32)
  ret i64 %v, !dbg !32
}

; IRT-LABEL: name: stack
; IRT:      DBG_VALUE %fixed-stack.1, 0, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; IRT-NEXT: DBG_VALUE %fixed-stack.0, 0, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 32, 32)
define hidden i64 @stack(<32 x i32> %pad, i64 %v) !dbg !40 {
    #dbg_value(i64 %v, !41, !DIExpression(), !42)
  ret i64 %v, !dbg !42
}

; Nested merges are split down to their sources.
; IRT-LABEL: name: nested
; IRT:      [[P0:%[0-9]+]]:_(i32) = COPY $vgpr0
; IRT-NEXT: [[P1:%[0-9]+]]:_(i32) = COPY $vgpr1
; IRT-NEXT: [[P2:%[0-9]+]]:_(i32) = COPY $vgpr2
; IRT-NEXT: [[P3:%[0-9]+]]:_(i32) = COPY $vgpr3
; IRT:      DBG_VALUE [[P0]](i32), $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; IRT-NEXT: DBG_VALUE [[P1]](i32), $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; IRT-NEXT: DBG_VALUE [[P2]](i32), $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 64, 32)
; IRT-NEXT: DBG_VALUE [[P3]](i32), $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 96, 32)
define hidden <2 x double> @nested(<2 x double> %v) !dbg !50 {
    #dbg_value(<2 x double> %v, !51, !DIExpression(), !52)
  ret <2 x double> %v, !dbg !52
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !{})
!4 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!5 = !DIBasicType(name: "int128", size: 128, encoding: DW_ATE_signed)
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 128, elements: !{!8})
!7 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!8 = !DISubrange(count: 2)
!10 = distinct !DISubprogram(name: "in_fragment", type: !3, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DILocalVariable(name: "v", arg: 1, scope: !10, type: !5)
!12 = !DILocation(line: 1, scope: !10)
!20 = distinct !DISubprogram(name: "clipped", type: !3, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DILocalVariable(name: "v", arg: 1, scope: !20, type: !4)
!22 = !DILocation(line: 1, scope: !20)
!30 = distinct !DISubprogram(name: "reg_and_stack", type: !3, spFlags: DISPFlagDefinition, unit: !0)
!31 = !DILocalVariable(name: "v", arg: 2, scope: !30, type: !4)
!32 = !DILocation(line: 1, scope: !30)
!40 = distinct !DISubprogram(name: "stack", type: !3, spFlags: DISPFlagDefinition, unit: !0)
!41 = !DILocalVariable(name: "v", arg: 2, scope: !40, type: !4)
!42 = !DILocation(line: 1, scope: !40)
!50 = distinct !DISubprogram(name: "nested", type: !3, spFlags: DISPFlagDefinition, unit: !0)
!51 = !DILocalVariable(name: "v", arg: 1, scope: !50, type: !6)
!52 = !DILocation(line: 1, scope: !50)
