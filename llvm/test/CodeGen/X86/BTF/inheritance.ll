; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s
; CHECK: BTF_KIND_STRUCT(id = 1)
; CHECK-NEXT: .long 67108866
; CHECK-NEXT: .long 8
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long 2
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long {{[0-9]+}}
; CHECK-NEXT: .long 3
; CHECK-NEXT: .long 32
; D keeps its non-virtual base. V (virtual base) and W (base V) have no fixed
; layout: neither they nor their variables are emitted.
; CHECK-NOT: BTF_KIND_FWD
; CHECK:     BTF_KIND_VAR(id = 4)
; CHECK-NOT: BTF_KIND_{{VAR|FWD}}

@d = global i64 0, !dbg !0
@v = global i64 0, !dbg !12
@w = global i64 0, !dbg !20
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "d", scope: !2, file: !3, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, emissionKind: FullDebug, globals: !11)
!3 = !DIFile(filename: "test.cc", directory: "/")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "D", file: !3, size: 64, elements: !5)
!5 = !{!6, !7}
!6 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !4, baseType: !14)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !4, baseType: !16, size: 32, offset: 32)
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 7, !"BTF", i32 1}
!11 = !{!0, !12, !20}
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "v", scope: !2, file: !3, type: !17, isLocal: false, isDefinition: true)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !3, size: 32, elements: !15)
!15 = !{}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DICompositeType(tag: DW_TAG_structure_type, name: "V", file: !3, size: 64, elements: !18)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !17, baseType: !14, flags: DIFlagVirtual)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "w", scope: !2, file: !3, type: !22, isLocal: false, isDefinition: true)
!22 = !DICompositeType(tag: DW_TAG_structure_type, name: "W", file: !3, size: 64, elements: !23)
!23 = !{!24}
!24 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !22, baseType: !17)
