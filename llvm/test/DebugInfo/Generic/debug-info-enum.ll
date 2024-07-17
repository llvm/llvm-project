; Test enumeration representation in DWARF debug info:
; * test value representation for each possible underlying integer type
; * test the integer type is as expected
; * test the DW_AT_enum_class attribute is present (resp. absent) as expected.
; * test that DW_AT_type is present for v3 and greater, and v2 when strict DWARF
;   is not enabled.

; RUN: llc -debugger-tune=gdb -dwarf-version=4 -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s --check-prefixes=CHECK,CHECK-DW4,CHECK-TYPE
; RUN: llc -debugger-tune=gdb -dwarf-version=3 -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s --check-prefixes=CHECK,CHECK-TYPE
; RUN: llc -debugger-tune=gdb -dwarf-version=2 -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s --check-prefixes=CHECK,CHECK-TYPE
; RUN: llc -debugger-tune=gdb -dwarf-version=2 -strict-dwarf=true -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s --check-prefixes=CHECK,CHECK-DW2-STRICT

@x0 = global i8 0, align 1, !dbg !0
@x1 = global i8 0, align 1, !dbg !46
@x2 = global i16 0, align 2, !dbg !48
@x3 = global i16 0, align 2, !dbg !50
@x4 = global i32 0, align 4, !dbg !52
@x5 = global i32 0, align 4, !dbg !54
@x6 = global i64 0, align 8, !dbg !56
@x7 = global i64 0, align 8, !dbg !58
@x8 = global i32 0, align 4, !dbg !60

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!62}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x0", scope: !2, file: !3, line: 5, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (/data/src/llvm/tools/clang 0c08d9830124a75675348b4eeb47256f3da6693d) (/data/src/llvm cf29510f52faa77b98510cd53276f564d1f4f41f)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !45)
!3 = !DIFile(filename: "/data/src/llvm-dev/tools/clang/test/CodeGen/debug-info-enum.cpp", directory: "/work/build/clang-dev")
!4 = !{!5, !10, !14, !19, !23, !28, !32, !37, !41}

; Test enumeration with a fixed "signed char" underlying type.
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E0", file: !3, line: 2, baseType: !6, size: 8, flags: DIFlagEnumClass, elements: !7, identifier: "_ZTS2E0")
!6 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!7 = !{!8, !9}
!8 = !DIEnumerator(name: "A0", value: -128)
!9 = !DIEnumerator(name: "B0", value: 127)
; CHECK:         DW_TAG_enumeration_type
; CHECK-DW2-STRICT-NOT: DW_AT_type
; CHECK-TYPE:      DW_AT_type{{.*}}"signed char"
; CHECK-DW4:       DW_AT_enum_class        (true)
; CHECK:           DW_AT_name      ("E0")
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("A0")
; CHECK-NEXT:      DW_AT_const_value     (-128)
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("B0")
; CHECK-NEXT:      DW_AT_const_value     (127)

; Test enumeration with a fixed "unsigned char" underlying type.
!10 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E1", file: !3, line: 12, baseType: !11, size: 8, flags: DIFlagEnumClass, elements: !12, identifier: "_ZTS2E1")
!11 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!12 = !{!13}
!13 = !DIEnumerator(name: "A1", value: 255, isUnsigned: true)
; CHECK:         DW_TAG_enumeration_type
; CHECK-DW2-STRICT-NOT: DW_AT_type
; CHECK-TYPE:      DW_AT_type{{.*}}"unsigned char"
; CHECK-DW4:       DW_AT_enum_class        (true)
; CHECK:           DW_AT_name      ("E1")
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("A1")
; CHECK-NEXT:      DW_AT_const_value     (255)

; Test enumeration with a fixed "short" underlying type.
!14 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E2", file: !3, line: 18, baseType: !15, size: 16, flags: DIFlagEnumClass, elements: !16, identifier: "_ZTS2E2")
!15 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!16 = !{!17, !18}
!17 = !DIEnumerator(name: "A2", value: -32768)
!18 = !DIEnumerator(name: "B2", value: 32767)
; CHECK:         DW_TAG_enumeration_type
; CHECK-DW2-STRICT-NOT:   DW_AT_type
; CHECK-TYPE:      DW_AT_type{{.*}} "short"
; CHECK-DW4:       DW_AT_enum_class        (true)
; CHECK:           DW_AT_name      ("E2")
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("A2")
; CHECK-NEXT:      DW_AT_const_value     (-32768)
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("B2")
; CHECK-NEXT:      DW_AT_const_value     (32767)

; Test enumeration with a fixed "unsigned short" underlying type.
!19 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E3", file: !3, line: 28, baseType: !20, size: 16, flags: DIFlagEnumClass, elements: !21, identifier: "_ZTS2E3")
!20 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!21 = !{!22}
!22 = !DIEnumerator(name: "A3", value: 65535, isUnsigned: true)
; CHECK:         DW_TAG_enumeration_type
; CHECK-DW2-STRICT-NOT: DW_AT_type
; CHECK-TYPE       DW_AT_type{{.*}}"unsigned short"
; CHECK-DW4:       DW_AT_enum_class        (true)
; CHECK:           DW_AT_name      ("E3")
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("A3")
; CHECK-NEXT:      DW_AT_const_value     (65535)

; Test enumeration with a fixed "int" underlying type.
!23 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E4", file: !3, line: 34, baseType: !24, size: 32, flags: DIFlagEnumClass, elements: !25, identifier: "_ZTS2E4")
!24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!25 = !{!26, !27}
!26 = !DIEnumerator(name: "A4", value: -2147483648)
!27 = !DIEnumerator(name: "B4", value: 2147483647)
; CHECK:         DW_TAG_enumeration_type
; CHECK-DW2-STRICT-NOT: DW_AT_type
; CHECK-TYPE:      DW_AT_type{{.*}}"int"
; CHECK-DW4:       DW_AT_enum_class        (true)
; CHECK:           DW_AT_name      ("E4")
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("A4")
; CHECK-NEXT:      DW_AT_const_value     (-2147483648)
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("B4")
; CHECK-NEXT:      DW_AT_const_value     (2147483647)

; Test enumeration with a fixed "unsigend int" underlying type.
!28 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E5", file: !3, line: 41, baseType: !29, size: 32, flags: DIFlagEnumClass, elements: !30, identifier: "_ZTS2E5")
!29 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!30 = !{!31}
!31 = !DIEnumerator(name: "A5", value: 4294967295, isUnsigned: true)
; CHECK:         DW_TAG_enumeration_type
; CHECK-DW2-STRICT-NOT: DW_AT_type
; CHECK-TYPE:      DW_AT_type{{.*}}"unsigned int"
; CHECK-DW4:       DW_AT_enum_class        (true)
; CHECK:           DW_AT_name      ("E5")
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("A5")
; CHECK-NEXT:      DW_AT_const_value     (4294967295)

; Test enumeration with a fixed "long long" underlying type.
!32 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E6", file: !3, line: 47, baseType: !33, size: 64, flags: DIFlagEnumClass, elements: !34, identifier: "_ZTS2E6")
!33 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!34 = !{!35, !36}
!35 = !DIEnumerator(name: "A6", value: -9223372036854775808)
!36 = !DIEnumerator(name: "B6", value: 9223372036854775807)
; CHECK:         DW_TAG_enumeration_type
; CHECK-DW2-STRICT-NOT:   DW_AT_type
; CHECK-TYPE:      DW_AT_type{{.*}}"long long int"
; CHECK-DW4:       DW_AT_enum_class        (true)
; CHECK:           DW_AT_name      ("E6")
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("A6")
; CHECK-NEXT:      DW_AT_const_value     (-9223372036854775808)
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("B6")
; CHECK-NEXT:      DW_AT_const_value     (9223372036854775807)

; Test enumeration with a fixed "unsigned long long" underlying type.
!37 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E7", file: !3, line: 57, baseType: !38, size: 64, flags: DIFlagEnumClass, elements: !39, identifier: "_ZTS2E7")
!38 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!39 = !{!40}
!40 = !DIEnumerator(name: "A7", value: 18446744073709551615, isUnsigned: true)
; CHECK:         DW_TAG_enumeration_type
; CHECK-DW2-STRICT-NOT:   DW_AT_type
; CHECK-TYPE:      DW_AT_type{{.*}}"long long unsigned int"
; CHECK-DW4:       DW_AT_enum_class        (true)
; CHECK:           DW_AT_name      ("E7")
; CHECK:         DW_TAG_enumerator
; CHECK:           DW_AT_name    ("A7")
; CHECK-NEXT:      DW_AT_const_value     (18446744073709551615)

; Test enumeration without a fixed underlying type. The underlying type should
; still be present (for DWARF >= 3), but the DW_AT_enum_class attribute should
; be absent.
!41 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E8", file: !3, line: 63, baseType: !24, size: 32, elements: !42, identifier: "_ZTS2E8")
!42 = !{!43, !44}
!43 = !DIEnumerator(name: "A8", value: -128)
!44 = !DIEnumerator(name: "B8", value: 127)
; CHECK:         DW_TAG_enumeration_type
; CHECK-DW2-STRICT-NOT:   DW_AT_type
; CHECK-TYPE:      DW_AT_type{{.*}}"int"
; CHECK-NOT:       DW_AT_enum_class
; CHECK:           DW_AT_name      ("E8")

; Test enumeration without a fixed underlying type, but with the DIFlagEnumClass
; set. The DW_AT_enum_class attribute should be absent. This behaviour is
; intented to keep compatibilty with existing DWARF consumers, which may imply
; the type is present whenever DW_AT_enum_class is set.
!63 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E9", file: !3, line: 63, size: 32, flags: DIFlagEnumClass,  elements: !64, identifier: "_ZTS2E9")
!64 = !{!65, !66}
!65 = !DIEnumerator(name: "A9", value: -128)
!66 = !DIEnumerator(name: "B9", value: 127)
; CHECK:     DW_TAG_enumeration_type
; CHECK-NOT:   DW_AT_type
; CHECK-NOT:   DW_AT_enum_class
; CHECK:       DW_AT_name      ("E9")

!45 = !{!0, !46, !48, !50, !52, !54, !56, !58, !60, !67}
!46 = !DIGlobalVariableExpression(var: !47, expr: !DIExpression())
!47 = distinct !DIGlobalVariable(name: "x1", scope: !2, file: !3, line: 12, type: !10, isLocal: false, isDefinition: true)
!48 = !DIGlobalVariableExpression(var: !49, expr: !DIExpression())
!49 = distinct !DIGlobalVariable(name: "x2", scope: !2, file: !3, line: 21, type: !14, isLocal: false, isDefinition: true)
!50 = !DIGlobalVariableExpression(var: !51, expr: !DIExpression())
!51 = distinct !DIGlobalVariable(name: "x3", scope: !2, file: !3, line: 28, type: !19, isLocal: false, isDefinition: true)
!52 = !DIGlobalVariableExpression(var: !53, expr: !DIExpression())
!53 = distinct !DIGlobalVariable(name: "x4", scope: !2, file: !3, line: 34, type: !23, isLocal: false, isDefinition: true)
!54 = !DIGlobalVariableExpression(var: !55, expr: !DIExpression())
!55 = distinct !DIGlobalVariable(name: "x5", scope: !2, file: !3, line: 41, type: !28, isLocal: false, isDefinition: true)
!56 = !DIGlobalVariableExpression(var: !57, expr: !DIExpression())
!57 = distinct !DIGlobalVariable(name: "x6", scope: !2, file: !3, line: 50, type: !32, isLocal: false, isDefinition: true)
!58 = !DIGlobalVariableExpression(var: !59, expr: !DIExpression())
!59 = distinct !DIGlobalVariable(name: "x7", scope: !2, file: !3, line: 57, type: !37, isLocal: false, isDefinition: true)
!60 = !DIGlobalVariableExpression(var: !61, expr: !DIExpression())
!61 = distinct !DIGlobalVariable(name: "x8", scope: !2, file: !3, line: 63, type: !41, isLocal: false, isDefinition: true)
!67 = !DIGlobalVariableExpression(var: !68, expr: !DIExpression())
!68 = distinct !DIGlobalVariable(name: "x9", scope: !2, file: !3, line: 63, type: !63, isLocal: false, isDefinition: true)
!62 = !{i32 2, !"Debug Info Version", i32 3}
