; RUN: llc --filetype=obj %s -o %t.dxbc
; RUN: llvm-objcopy --dump-section=ILDB=%t.bc %t.dxbc
; RUN: dxil-dis %t.bc -o - | FileCheck %s

target triple = "dxil-unknown-shadermodel6.3-library"

; CHECK-NOT: call void @llvm.dbg.value(

define void @fn() !dbg !17 {
entry:
    #dbg_value(!DIArgList(ptr poison, i32 poison), !21, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), !23)
  ret void, !dbg !23
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{}
!3 = !{!4, !13}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(name: "s", scope: !0, file: !1, line: 1, type: !6, isLocal: true, isDefinition: true)
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, elements: !11)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 1, size: 32, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !7, file: !1, line: 1, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 1)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "idx", scope: !0, file: !1, line: 2, type: !10, isLocal: false, isDefinition: true)
!15 = !{i32 7, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = distinct !DISubprogram(name: "fn", scope: !1, file: !1, line: 4, type: !18, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !{!21}
!21 = !DILocalVariable(name: "local", scope: !17, file: !1, line: 5, type: !22)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!23 = !DILocation(line: 5, scope: !17)
