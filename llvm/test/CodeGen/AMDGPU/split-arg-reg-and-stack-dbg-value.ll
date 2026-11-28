; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck %s

; Arguments split between registers and the stack are described by a fragment
; per register and a fragment per stack slot.

; CHECK-LABEL: {{^}}split_i64:
; CHECK:      ;DEBUG_VALUE: split_i64:v <- [DW_OP_LLVM_fragment 0 32] $vgpr30
; CHECK:      ;DEBUG_VALUE: split_i64:v <- [DW_OP_LLVM_fragment 32 32] [$sgpr32+0]
define i64 @split_i64(<30 x i32> %pad, i64 %v) !dbg !10 {
  #dbg_value(i64 %v, !11, !DIExpression(), !12)
  %r = add i64 %v, 1, !dbg !12
  ret i64 %r, !dbg !12
}

; CHECK-LABEL: {{^}}split_v4i32:
; CHECK:      ;DEBUG_VALUE: split_v4i32:v <- [DW_OP_LLVM_fragment 32 32] $vgpr30
; CHECK:      ;DEBUG_VALUE: split_v4i32:v <- [DW_OP_LLVM_fragment 0 32] $vgpr29
; CHECK:      ;DEBUG_VALUE: split_v4i32:v <- [DW_OP_LLVM_fragment 64 32] [$sgpr32+0]
; CHECK:      ;DEBUG_VALUE: split_v4i32:v <- [DW_OP_plus_uconst 4, DW_OP_LLVM_fragment 96 32] [$sgpr32+0]
define <4 x i32> @split_v4i32(<29 x i32> %pad, <4 x i32> %v) !dbg !20 {
  #dbg_value(<4 x i32> %v, !21, !DIExpression(), !22)
  %r = add <4 x i32> %v, splat (i32 1), !dbg !22
  ret <4 x i32> %r, !dbg !22
}

; Only the bits inside the expression's fragment are described.
; CHECK-LABEL: {{^}}split_in_fragment:
; CHECK:      ;DEBUG_VALUE: split_in_fragment:v <- [DW_OP_LLVM_fragment 0 32] $vgpr30
; CHECK:      ;DEBUG_VALUE: split_in_fragment:v <- [DW_OP_LLVM_fragment 32 16] [$sgpr32+0]
define i64 @split_in_fragment(<30 x i32> %pad, i64 %v) !dbg !30 {
  #dbg_value(i64 %v, !31, !DIExpression(DW_OP_LLVM_fragment, 0, 48), !32)
  %r = add i64 %v, 1, !dbg !32
  ret i64 %r, !dbg !32
}

; Unused arguments are described by all of their parts too.
; CHECK-LABEL: {{^}}split_unused:
; CHECK:      ;DEBUG_VALUE: split_unused:v <- [DW_OP_LLVM_fragment 0 32] $vgpr30
; CHECK:      ;DEBUG_VALUE: split_unused:v <- [DW_OP_LLVM_fragment 32 32] [$sgpr32+0]
define void @split_unused(<30 x i32> %pad, i64 %v) !dbg !40 {
  #dbg_value(i64 %v, !41, !DIExpression(), !42)
  ret void, !dbg !42
}

; CHECK-LABEL: {{^}}unused_in_regs:
; CHECK:      ;DEBUG_VALUE: unused_in_regs:v <- [DW_OP_LLVM_fragment 0 32] $vgpr0
; CHECK:      ;DEBUG_VALUE: unused_in_regs:v <- [DW_OP_LLVM_fragment 32 32] $vgpr1
define void @unused_in_regs(i64 %v) !dbg !50 {
  #dbg_value(i64 %v, !51, !DIExpression(), !52)
  ret void, !dbg !52
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !{})
!4 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 128, flags: DIFlagVector, elements: !{!7})
!7 = !DISubrange(count: 4)
!10 = distinct !DISubprogram(name: "split_i64", type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DILocalVariable(name: "v", arg: 2, scope: !10, type: !4)
!12 = !DILocation(line: 1, scope: !10)
!20 = distinct !DISubprogram(name: "split_v4i32", type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocalVariable(name: "v", arg: 2, scope: !20, type: !6)
!22 = !DILocation(line: 1, scope: !20)
!30 = distinct !DISubprogram(name: "split_in_fragment", type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILocalVariable(name: "v", arg: 2, scope: !30, type: !4)
!32 = !DILocation(line: 1, scope: !30)
!40 = distinct !DISubprogram(name: "split_unused", type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!41 = !DILocalVariable(name: "v", arg: 2, scope: !40, type: !4)
!42 = !DILocation(line: 1, scope: !40)
!50 = distinct !DISubprogram(name: "unused_in_regs", type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!51 = !DILocalVariable(name: "v", arg: 1, scope: !50, type: !4)
!52 = !DILocation(line: 1, scope: !50)
