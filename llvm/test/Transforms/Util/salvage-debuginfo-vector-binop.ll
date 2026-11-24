; RUN: opt -passes=dce %s -S -o - | FileCheck %s
; RUN: opt -passes=dce -use-constant-int-for-fixed-length-splat %s -S -o - | FileCheck %s

; DWARF expressions have no lane-wise arithmetic, so dead vector binary
; operators must not be salvaged into scalar expressions.

; CHECK-LABEL: define void @f(
; CHECK-NEXT:    #dbg_value(<2 x i32> poison, ![[#]], !DIExpression(),
; CHECK-NEXT:    #dbg_value(<2 x i32> poison, ![[#]], !DIExpression(),
; CHECK-NEXT:    #dbg_value(<2 x i32> poison, ![[#]], !DIExpression(),
; CHECK-NEXT:    #dbg_value(i32 %s, ![[#]], !DIExpression(DW_OP_plus_uconst, 5, DW_OP_stack_value),
; CHECK-NEXT:    ret void
define void @f(<2 x i32> %a, <2 x i32> %b, i32 %s) !dbg !5 {
  %r = add <2 x i32> %a, %b
    #dbg_value(<2 x i32> %r, !7, !DIExpression(), !9)
  %c = add <2 x i32> %a, splat (i32 3)
    #dbg_value(<2 x i32> %c, !7, !DIExpression(), !9)
  %d = add <2 x i32> %a, <i32 1, i32 2>
    #dbg_value(<2 x i32> %d, !7, !DIExpression(), !9)
  %e = add i32 %s, 5
    #dbg_value(i32 %e, !10, !DIExpression(), !9)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !6, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(types: !{})
!7 = !DILocalVariable(name: "v", scope: !5, file: !1, type: !8)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 64, flags: DIFlagVector, elements: !{!12})
!9 = !DILocation(line: 1, scope: !5)
!10 = !DILocalVariable(name: "s", scope: !5, file: !1, type: !11)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DISubrange(count: 2)
