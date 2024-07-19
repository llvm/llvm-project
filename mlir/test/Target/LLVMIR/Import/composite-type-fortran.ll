; RUN: mlir-translate -import-llvm -mlir-print-debuginfo  %s | FileCheck %s

define void @fn_with_composite() !dbg !3 {
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !2, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "test.f90", directory: "")
!3 = distinct !DISubprogram(name: "fn_with_composite", scope: !1, file: !2, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
!5 = !{null, !6}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, elements: !8, dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), associated: !DIExpression(DW_OP_lit0, DW_OP_eq), allocated: !DIExpression(DW_OP_lit0, DW_OP_ne), rank: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 16, DW_OP_deref))
!7 = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 5, lowerBound: 1)
!10 = !DILocation(line: 26, column: 3, scope: !3)

; CHECK: #llvm.di_composite_type<tag = DW_TAG_array_type,
; CHECK-SAME: dataLocation = <[DW_OP_push_object_address, DW_OP_deref]>
; CHECK-SAME: rank = <[DW_OP_push_object_address, DW_OP_plus_uconst(16), DW_OP_deref]>
; CHECK-SAME: allocated = <[DW_OP_lit0, DW_OP_ne]>
; CHECK-SAME: associated = <[DW_OP_lit0, DW_OP_eq]>
