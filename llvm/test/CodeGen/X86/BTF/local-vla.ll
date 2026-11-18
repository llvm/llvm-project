; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s --implicit-check-not=BTF_KIND_ARRAY

; The inner dimension is constant, but neither array entry should be emitted.
; CHECK: .section .BTF
; CHECK: BTF_KIND_FUNC(

define void @f(i32 %n, ptr %a) !dbg !4 {
  #dbg_value(i32 %n, !7, !DIExpression(), !8)
  #dbg_declare(ptr %a, !9, !DIExpression(), !8)
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "vla.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"BTF", i32 1}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, type: !5, unit: !0, spFlags: DISPFlagDefinition)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !12, !13}
!7 = !DILocalVariable(name: "n", arg: 1, scope: !4, type: !12)
!8 = !DILocation(line: 1, scope: !4)
!9 = !DILocalVariable(name: "a", scope: !4, type: !10)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, elements: !11)
!11 = !{!DISubrange(count: !7), !DISubrange(count: 4)}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
