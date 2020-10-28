; Test that the debug info for the vector type is correctly codegenerated
; when the DISubrange has no count, but only an upperbound.
; RUN: llc -mtriple aarch64 -mattr=+sve -filetype=obj -o %t %s
; RUN: llvm-dwarfdump %t | FileCheck %s
; RUN: rm %t

; CHECK:      {{.*}}: DW_TAG_subrange_type
; CHECK-NEXT:   DW_AT_type    ({{.*}} "__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:   DW_AT_upper_bound     (DW_OP_lit8, DW_OP_bregx VG+0, DW_OP_mul, DW_OP_lit1, DW_OP_minus)

define <vscale x 16 x i8> @test_svint8_t(<vscale x 16 x i8> returned %op1) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata <vscale x 16 x i8> %op1, metadata !19, metadata !DIExpression()), !dbg !20
  ret <vscale x 16 x i8> %op1, !dbg !21
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "dbg-sve-types.ll", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0"}
!7 = distinct !DISubprogram(name: "test_svint8_t", scope: !8, file: !8, line: 5, type: !9, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!8 = !DIFile(filename: "dbg-sve-types.ll", directory: "")
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "svint8_t", file: !12, line: 32, baseType: !13)
!12 = !DIFile(filename: "lib/clang/12.0.0/include/arm_sve.h", directory: "")
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "__SVInt8_t", file: !1, baseType: !14)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, flags: DIFlagVector, elements: !16)
!15 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!16 = !{!17}
!17 = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 8, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
!18 = !{!19}
!19 = !DILocalVariable(name: "op1", arg: 1, scope: !7, file: !8, line: 5, type: !11)
!20 = !DILocation(line: 0, scope: !7)
!21 = !DILocation(line: 5, column: 39, scope: !7)
