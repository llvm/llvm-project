; RUN: llvm-as -disable-output %s 2>&1 | FileCheck %s --allow-empty

; CHECK-NOT: Subrange must contain count or upperBound

define void @fn_(ptr %0) {
  #dbg_declare(ptr %0, !12, !DIExpression(), !13)
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !2, emissionKind: FullDebug)
!2 = !DIFile(filename: "test.f90", directory: "")
!3 = distinct !DISubprogram(scope: !2, type: !5, unit: !1)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{!7, !8}
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, elements: !9)
!8 = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
!9 = !{!11}
!11 = !DISubrange()
!12 = !DILocalVariable(name: "a1", arg: 1, scope: !3, file: !2, type: !7)
!13 = !DILocation(line: 4, column: 3, scope: !3)
