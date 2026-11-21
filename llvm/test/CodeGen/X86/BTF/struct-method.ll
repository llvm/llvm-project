; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s
; A struct's non-layout method is ignored while its member is retained.
; CHECK-NOT: BTF_KIND_FUNC
; CHECK: BTF_KIND_STRUCT
; CHECK-NEXT: .long 67108865
; CHECK-NEXT: .long {{[0-9]+}}
; CHECK-NEXT: .long {{[0-9]+}}
; CHECK-NEXT: .long 2
; CHECK-NEXT: .long 0
; CHECK: BTF_KIND_VAR
; CHECK: BTF_KIND_VAR
; CHECK: .byte 67
; CHECK-NOT: .ascii "bad"
; CHECK: .ascii "pointer"

@c = global i32 0, !dbg !0
@complex = global [2 x double] zeroinitializer, !dbg !17
@bad = global [2 x double] zeroinitializer, !dbg !21
@pointer = global ptr null, !dbg !25
define void @f(double %x) !dbg !8 {
  ret void, !dbg !13
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !14}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "test.cc", directory: "/")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !3, line: 1, size: 32, elements: !15)
!5 = !{!0, !17, !21, !25}
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 2, type: !9, unit: !2, spFlags: DISPFlagDefinition)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !23, !20}
!11 = !DIBasicType(name: "decimal float", size: 128, encoding: DW_ATE_decimal_float)
!13 = !DILocation(line: 2, column: 1, scope: !8)
!14 = !{i32 7, !"BTF", i32 1}
!15 = !{!16, !19}
!16 = !DISubprogram(name: "method", scope: !4, file: !3, type: !9, spFlags: 0)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "complex", scope: !2, file: !3, line: 1, type: !11, isLocal: false, isDefinition: true)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "n", scope: !4, baseType: !20, size: 32)
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "bad", scope: !2, file: !3, type: !23, isLocal: false, isDefinition: true)
!23 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !24)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "bad_type", baseType: !11)
!25 = !DIGlobalVariableExpression(var: !26, expr: !DIExpression())
!26 = distinct !DIGlobalVariable(name: "pointer", scope: !2, file: !3, type: !27, isLocal: false, isDefinition: true)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !24, size: 64)
