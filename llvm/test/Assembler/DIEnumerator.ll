; Round-trip test for the following program:
; ```
; enum E0 {  A0 = -2147483648, B0 = 2147483647 } x0;
; enum E1 : signed int { A1 = -2147483648, B1 = 2147483647 } x1;
; enum E2 : signed long long { A2 = -9223372036854775807LL - 1,
;                              B2 = 9223372036854775807LL } x2;
; enum E3 : unsigned long long { A3 = 0x8000000000000000ULL } x3;
; ```
; Test FixedEnum flag presence/absence, the underlying integer type and
; enumerator values (signed and unsigned, and extreme cases) all survive through
; the round-trip.

; RUN: llvm-as %s -o - | llvm-dis | llvm-as | llvm-dis | FileCheck %s

@x0 = global i32 0, align 4, !dbg !0
@x1 = global i32 0, align 4, !dbg !24
@x2 = global i64 0, align 8, !dbg !26
@x3 = global i64 0, align 8, !dbg !28

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!30, !31, !32}
!llvm.ident = !{!33}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x0", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (/data/src/llvm/tools/clang 697b0cb4c2e712a28767c2f7fe50c90bae7255f5) (/data/src/llvm 5ba8dcca7470b5da405bc92b9681b1f36e5d6772)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !23)
!3 = !DIFile(filename: "e.cc", directory: "/work/build/clang-dev")
!4 = !{!5, !10, !14, !19}


!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E0", file: !3, line: 1, baseType: !6, size: 32, elements: !7, identifier: "_ZTS2E0")
; CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E0"{{.*}}, baseType: ![[INT:[0-9]+]]
; CHECK-NOT: FixedEnum
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; CHECK: ![[INT]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!8, !9}
!8 = !DIEnumerator(name: "A0", value: -2147483648)
!9 = !DIEnumerator(name: "B0", value: 2147483647)
; CHECK: !DIEnumerator(name: "A0", value: -2147483648)
; CHECK: !DIEnumerator(name: "B0", value: 2147483647)


!10 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E1", file: !3, line: 3, baseType: !6, size: 32, flags: DIFlagFixedEnum, elements: !11, identifier: "_ZTS2E1")
; CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E1"{{.*}}, baseType: ![[INT]]
; CHECK-SAME: DIFlagFixedEnum
!11 = !{!12, !13}
!12 = !DIEnumerator(name: "A1", value: -2147483648)
!13 = !DIEnumerator(name: "B1", value: 2147483647)
; CHECK: !DIEnumerator(name: "A1", value: -2147483648)
; CHECK: !DIEnumerator(name: "B1", value: 2147483647)


!14 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E2", file: !3, line: 5, baseType: !15, size: 64, flags: DIFlagFixedEnum, elements: !16, identifier: "_ZTS2E2")
; CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E2"{{.*}}, baseType: ![[LONG:[0-9]+]]
; CHECK-SAME: DIFlagFixedEnum
!15 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
; CHECK: ![[LONG]] = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!16 = !{!17, !18}
!17 = !DIEnumerator(name: "A2", value: -9223372036854775808)
!18 = !DIEnumerator(name: "B2", value: 9223372036854775807)
; CHECK: !DIEnumerator(name: "A2", value: -9223372036854775808)
; CHECK: !DIEnumerator(name: "B2", value: 9223372036854775807)


!19 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E3", file: !3, line: 7, baseType: !20, size: 64, flags: DIFlagFixedEnum, elements: !21, identifier: "_ZTS2E3")
; CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "E3"{{.*}}, baseType: ![[ULONG:[0-9]+]]
; CHECK-SAME: DIFlagFixedEnum
!20 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
; CHECK: ![[ULONG]] = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!21 = !{!22}
!22 = !DIEnumerator(name: "A3", value: 9223372036854775808, isUnsigned: true)
; CHECK: !DIEnumerator(name: "A3", value: 9223372036854775808, isUnsigned: true)


!23 = !{!0, !24, !26, !28}
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression())
!25 = distinct !DIGlobalVariable(name: "x1", scope: !2, file: !3, line: 3, type: !10, isLocal: false, isDefinition: true)
!26 = !DIGlobalVariableExpression(var: !27, expr: !DIExpression())
!27 = distinct !DIGlobalVariable(name: "x2", scope: !2, file: !3, line: 5, type: !14, isLocal: false, isDefinition: true)
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = distinct !DIGlobalVariable(name: "x3", scope: !2, file: !3, line: 7, type: !19, isLocal: false, isDefinition: true)
!30 = !{i32 2, !"Dwarf Version", i32 4}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 1, !"wchar_size", i32 4}
!33 = !{!"clang version 7.0.0 (/data/src/llvm/tools/clang 697b0cb4c2e712a28767c2f7fe50c90bae7255f5) (/data/src/llvm 5ba8dcca7470b5da405bc92b9681b1f36e5d6772)"}
