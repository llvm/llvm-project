; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj - --codeview | FileCheck %s

; CHECK:     CodeViewTypes [

; CHECK:        FieldList (0x1003) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 0
; CHECK-NEXT:       Name: Min
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 255
; CHECK-NEXT:       Name: Max
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Enum (0x1004) {
; CHECK-NEXT:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:     NumEnumerators: 2
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     UnderlyingType: unsigned char (0x20)
; CHECK-NEXT:     FieldListType: <field list> (0x1003)
; CHECK-NEXT:     Name: U8Enum
; CHECK-NEXT:     LinkageName: .?AW4U8Enum@@
; CHECK-NEXT:   }

; CHECK:        FieldList (0x1007) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: -128
; CHECK-NEXT:       Name: Min
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 127
; CHECK-NEXT:       Name: Max
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Enum (0x1008) {
; CHECK-NEXT:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:     NumEnumerators: 2
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     UnderlyingType: char (0x70)
; CHECK-NEXT:     FieldListType: <field list> (0x1007)
; CHECK-NEXT:     Name: I8Enum
; CHECK-NEXT:     LinkageName: .?AW4I8Enum@@
; CHECK-NEXT:   }

; CHECK:        FieldList (0x100A) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 0
; CHECK-NEXT:       Name: Min
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 65535
; CHECK-NEXT:       Name: Max
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Enum (0x100B) {
; CHECK-NEXT:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:     NumEnumerators: 2
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     UnderlyingType: unsigned short (0x21)
; CHECK-NEXT:     FieldListType: <field list> (0x100A)
; CHECK-NEXT:     Name: U16Enum
; CHECK-NEXT:     LinkageName: .?AW4U16Enum@@
; CHECK-NEXT:   }

; CHECK:        FieldList (0x100D) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: -32768
; CHECK-NEXT:       Name: Min
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 32767
; CHECK-NEXT:       Name: Max
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Enum (0x100E) {
; CHECK-NEXT:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:     NumEnumerators: 2
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     UnderlyingType: short (0x11)
; CHECK-NEXT:     FieldListType: <field list> (0x100D)
; CHECK-NEXT:     Name: I16Enum
; CHECK-NEXT:     LinkageName: .?AW4I16Enum@@
; CHECK-NEXT:   }

; CHECK:        FieldList (0x1010) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 0
; CHECK-NEXT:       Name: Min
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 4294967294
; CHECK-NEXT:       Name: MaxMinusOne
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 4294967295
; CHECK-NEXT:       Name: Max
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Enum (0x1011) {
; CHECK-NEXT:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:     NumEnumerators: 3
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     UnderlyingType: unsigned (0x75)
; CHECK-NEXT:     FieldListType: <field list> (0x1010)
; CHECK-NEXT:     Name: U32Enum
; CHECK-NEXT:     LinkageName: .?AW4U32Enum@@
; CHECK-NEXT:   }

; CHECK:        FieldList (0x1013) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: -2147483648
; CHECK-NEXT:       Name: Min
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 2147483647
; CHECK-NEXT:       Name: Max
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Enum (0x1014) {
; CHECK-NEXT:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:     NumEnumerators: 2
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     UnderlyingType: int (0x74)
; CHECK-NEXT:     FieldListType: <field list> (0x1013)
; CHECK-NEXT:     Name: I32Enum
; CHECK-NEXT:     LinkageName: .?AW4I32Enum@@
; CHECK-NEXT:   }

; CHECK:        FieldList (0x1016) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 0
; CHECK-NEXT:       Name: Min
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 18446744073709551614
; CHECK-NEXT:       Name: MaxMinusOne
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 18446744073709551615
; CHECK-NEXT:       Name: Max
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Enum (0x1017) {
; CHECK-NEXT:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:     NumEnumerators: 3
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     UnderlyingType: unsigned __int64 (0x23)
; CHECK-NEXT:     FieldListType: <field list> (0x1016)
; CHECK-NEXT:     Name: U64Enum
; CHECK-NEXT:     LinkageName: .?AW4U64Enum@@
; CHECK-NEXT:   }

; CHECK:        FieldList (0x1019) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: -9223372036854775808
; CHECK-NEXT:       Name: Min
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: -9223372036854775807
; CHECK-NEXT:       Name: MinPlusOne
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 9223372036854775806
; CHECK-NEXT:       Name: MaxMinusOne
; CHECK-NEXT:     }
; CHECK-NEXT:     Enumerator {
; CHECK-NEXT:       TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       EnumValue: 9223372036854775807
; CHECK-NEXT:       Name: Max
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Enum (0x101A) {
; CHECK-NEXT:     TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:     NumEnumerators: 4
; CHECK-NEXT:     Properties [ (0x200)
; CHECK-NEXT:       HasUniqueName (0x200)
; CHECK-NEXT:     ]
; CHECK-NEXT:     UnderlyingType: __int64 (0x13)
; CHECK-NEXT:     FieldListType: <field list> (0x1019)
; CHECK-NEXT:     Name: I64Enum
; CHECK-NEXT:     LinkageName: .?AW4I64Enum@@
; CHECK-NEXT:   }

; Generated and reduced from the following C++ source:
; enum class U8Enum : unsigned char {
;   Min = 0,
;   Max = 255,
; };
; enum class I8Enum : char {
;   Min = -128,
;   Max = 127,
; };
; enum class U16Enum : unsigned short {
;   Min = 0,
;   Max = 65535,
; };
; enum class I16Enum : short {
;   Min = -32768,
;   Max = 32767,
; };
; enum class U32Enum : unsigned {
;   Min = 0,
;   MaxMinusOne = 4294967294,
;   Max = 4294967295,
; };
; enum class I32Enum : int {
;   Min = -2147483648,
;   Max = 2147483647,
; };
; enum class U64Enum : unsigned long long {
;   Min = 0,
;   MaxMinusOne = 18446744073709551614ULL,
;   Max = 18446744073709551615ULL,
; };
; enum class I64Enum : long long {
;   Min = -9223372036854775807LL - 1,
;   MinPlusOne = -9223372036854775807LL,
;   MaxMinusOne = 9223372036854775806LL,
;   Max = 9223372036854775807LL,
; };
; int main(){
;   auto u8 = U8Enum::Max;
;   auto i8 = I8Enum::Max;
;   auto u16 = U16Enum::Max;
;   auto i16 = I16Enum::Max;
;   auto u32 = U32Enum::Max;
;   auto i32 = I32Enum::Max;
;   auto u64 = U64Enum::Max;
;   auto i64 = I64Enum::Max;
;   return 0;
; }

; ModuleID = 'enum-limits.cpp'
source_filename = "enum-limits.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.51.36248"

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !72 {
entry:
  %retval = alloca i32, align 4
  %u8 = alloca i8, align 1
  %i8 = alloca i8, align 1
  %u16 = alloca i16, align 2
  %i16 = alloca i16, align 2
  %u32 = alloca i32, align 4
  %i32 = alloca i32, align 4
  %u64 = alloca i64, align 8
  %i64 = alloca i64, align 8
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %u8, !76, !DIExpression(), !77)
  store i8 -1, ptr %u8, align 1, !dbg !77
    #dbg_declare(ptr %i8, !78, !DIExpression(), !79)
  store i8 127, ptr %i8, align 1, !dbg !79
    #dbg_declare(ptr %u16, !80, !DIExpression(), !81)
  store i16 -1, ptr %u16, align 2, !dbg !81
    #dbg_declare(ptr %i16, !82, !DIExpression(), !83)
  store i16 32767, ptr %i16, align 2, !dbg !83
    #dbg_declare(ptr %u32, !84, !DIExpression(), !85)
  store i32 -1, ptr %u32, align 4, !dbg !85
    #dbg_declare(ptr %i32, !86, !DIExpression(), !87)
  store i32 2147483647, ptr %i32, align 4, !dbg !87
    #dbg_declare(ptr %u64, !88, !DIExpression(), !89)
  store i64 -1, ptr %u64, align 8, !dbg !89
    #dbg_declare(ptr %i64, !90, !DIExpression(), !91)
  store i64 9223372036854775807, ptr %i64, align 8, !dbg !91
  ret i32 0, !dbg !92
}

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.linker.options = !{!64, !65}
!llvm.module.flags = !{!66, !67, !68, !69, !70}
!llvm.ident = !{!71}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git (https://github.com/Nerixyz/llvm-project.git 9a2d9c7fef47c0af321150f75d8f21db06cccb43)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !47, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "enum-limits.cpp", directory: "F:\\Dev\\llvm-project\\llvm\\test\\DebugInfo\\COFF", checksumkind: CSK_MD5, checksum: "372fdab57a95194725a33026ed999ee0")
!2 = !{!3, !8, !13, !18, !23, !29, !34, !40}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "U8Enum", file: !1, line: 1, baseType: !4, size: 8, flags: DIFlagEnumClass, elements: !5, identifier: ".?AW4U8Enum@@")
!4 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!5 = !{!6, !7}
!6 = !DIEnumerator(name: "Min", value: 0, isUnsigned: true)
!7 = !DIEnumerator(name: "Max", value: 255, isUnsigned: true)
!8 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "I8Enum", file: !1, line: 5, baseType: !9, size: 8, flags: DIFlagEnumClass, elements: !10, identifier: ".?AW4I8Enum@@")
!9 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!10 = !{!11, !12}
!11 = !DIEnumerator(name: "Min", value: -128)
!12 = !DIEnumerator(name: "Max", value: 127)
!13 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "U16Enum", file: !1, line: 9, baseType: !14, size: 16, flags: DIFlagEnumClass, elements: !15, identifier: ".?AW4U16Enum@@")
!14 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!15 = !{!16, !17}
!16 = !DIEnumerator(name: "Min", value: 0, isUnsigned: true)
!17 = !DIEnumerator(name: "Max", value: 65535, isUnsigned: true)
!18 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "I16Enum", file: !1, line: 13, baseType: !19, size: 16, flags: DIFlagEnumClass, elements: !20, identifier: ".?AW4I16Enum@@")
!19 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!20 = !{!21, !22}
!21 = !DIEnumerator(name: "Min", value: -32768)
!22 = !DIEnumerator(name: "Max", value: 32767)
!23 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "U32Enum", file: !1, line: 17, baseType: !24, size: 32, flags: DIFlagEnumClass, elements: !25, identifier: ".?AW4U32Enum@@")
!24 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!25 = !{!26, !27, !28}
!26 = !DIEnumerator(name: "Min", value: 0, isUnsigned: true)
!27 = !DIEnumerator(name: "MaxMinusOne", value: 4294967294, isUnsigned: true)
!28 = !DIEnumerator(name: "Max", value: 4294967295, isUnsigned: true)
!29 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "I32Enum", file: !1, line: 22, baseType: !30, size: 32, flags: DIFlagEnumClass, elements: !31, identifier: ".?AW4I32Enum@@")
!30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!31 = !{!32, !33}
!32 = !DIEnumerator(name: "Min", value: -2147483648)
!33 = !DIEnumerator(name: "Max", value: 2147483647)
!34 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "U64Enum", file: !1, line: 26, baseType: !35, size: 64, flags: DIFlagEnumClass, elements: !36, identifier: ".?AW4U64Enum@@")
!35 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!36 = !{!37, !38, !39}
!37 = !DIEnumerator(name: "Min", value: 0, isUnsigned: true)
!38 = !DIEnumerator(name: "MaxMinusOne", value: 18446744073709551614, isUnsigned: true)
!39 = !DIEnumerator(name: "Max", value: 18446744073709551615, isUnsigned: true)
!40 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "I64Enum", file: !1, line: 31, baseType: !41, size: 64, flags: DIFlagEnumClass, elements: !42, identifier: ".?AW4I64Enum@@")
!41 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!42 = !{!43, !44, !45, !46}
!43 = !DIEnumerator(name: "Min", value: -9223372036854775808)
!44 = !DIEnumerator(name: "MinPlusOne", value: -9223372036854775807)
!45 = !DIEnumerator(name: "MaxMinusOne", value: 9223372036854775806)
!46 = !DIEnumerator(name: "Max", value: 9223372036854775807)
!47 = !{!48, !50, !52, !54, !56, !58, !60, !62}
!48 = !DIGlobalVariableExpression(var: !49, expr: !DIExpression(DW_OP_constu, 255, DW_OP_stack_value))
!49 = distinct !DIGlobalVariable(name: "Max", scope: !0, file: !1, line: 3, type: !3, isLocal: true, isDefinition: true)
!50 = !DIGlobalVariableExpression(var: !51, expr: !DIExpression(DW_OP_constu, 127, DW_OP_stack_value))
!51 = distinct !DIGlobalVariable(name: "Max", scope: !0, file: !1, line: 7, type: !8, isLocal: true, isDefinition: true)
!52 = !DIGlobalVariableExpression(var: !53, expr: !DIExpression(DW_OP_constu, 65535, DW_OP_stack_value))
!53 = distinct !DIGlobalVariable(name: "Max", scope: !0, file: !1, line: 11, type: !13, isLocal: true, isDefinition: true)
!54 = !DIGlobalVariableExpression(var: !55, expr: !DIExpression(DW_OP_constu, 32767, DW_OP_stack_value))
!55 = distinct !DIGlobalVariable(name: "Max", scope: !0, file: !1, line: 15, type: !18, isLocal: true, isDefinition: true)
!56 = !DIGlobalVariableExpression(var: !57, expr: !DIExpression(DW_OP_constu, 4294967295, DW_OP_stack_value))
!57 = distinct !DIGlobalVariable(name: "Max", scope: !0, file: !1, line: 20, type: !23, isLocal: true, isDefinition: true)
!58 = !DIGlobalVariableExpression(var: !59, expr: !DIExpression(DW_OP_constu, 2147483647, DW_OP_stack_value))
!59 = distinct !DIGlobalVariable(name: "Max", scope: !0, file: !1, line: 24, type: !29, isLocal: true, isDefinition: true)
!60 = !DIGlobalVariableExpression(var: !61, expr: !DIExpression(DW_OP_constu, 18446744073709551615, DW_OP_stack_value))
!61 = distinct !DIGlobalVariable(name: "Max", scope: !0, file: !1, line: 29, type: !34, isLocal: true, isDefinition: true)
!62 = !DIGlobalVariableExpression(var: !63, expr: !DIExpression(DW_OP_constu, 9223372036854775807, DW_OP_stack_value))
!63 = distinct !DIGlobalVariable(name: "Max", scope: !0, file: !1, line: 35, type: !40, isLocal: true, isDefinition: true)
!64 = !{!"/DEFAULTLIB:libcmt.lib"}
!65 = !{!"/DEFAULTLIB:oldnames.lib"}
!66 = !{i32 2, !"CodeView", i32 1}
!67 = !{i32 2, !"Debug Info Version", i32 3}
!68 = !{i32 8, !"PIC Level", i32 2}
!69 = !{i32 7, !"uwtable", i32 2}
!70 = !{i32 1, !"MaxTLSAlign", i32 65536}
!71 = !{!"clang version 24.0.0git (https://github.com/Nerixyz/llvm-project.git 9a2d9c7fef47c0af321150f75d8f21db06cccb43)"}
!72 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 38, type: !73, scopeLine: 38, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !75)
!73 = !DISubroutineType(types: !74)
!74 = !{!30}
!75 = !{}
!76 = !DILocalVariable(name: "u8", scope: !72, file: !1, line: 39, type: !3)
!77 = !DILocation(line: 39, scope: !72)
!78 = !DILocalVariable(name: "i8", scope: !72, file: !1, line: 40, type: !8)
!79 = !DILocation(line: 40, scope: !72)
!80 = !DILocalVariable(name: "u16", scope: !72, file: !1, line: 41, type: !13)
!81 = !DILocation(line: 41, scope: !72)
!82 = !DILocalVariable(name: "i16", scope: !72, file: !1, line: 42, type: !18)
!83 = !DILocation(line: 42, scope: !72)
!84 = !DILocalVariable(name: "u32", scope: !72, file: !1, line: 43, type: !23)
!85 = !DILocation(line: 43, scope: !72)
!86 = !DILocalVariable(name: "i32", scope: !72, file: !1, line: 44, type: !29)
!87 = !DILocation(line: 44, scope: !72)
!88 = !DILocalVariable(name: "u64", scope: !72, file: !1, line: 45, type: !34)
!89 = !DILocation(line: 45, scope: !72)
!90 = !DILocalVariable(name: "i64", scope: !72, file: !1, line: 46, type: !40)
!91 = !DILocation(line: 46, scope: !72)
!92 = !DILocation(line: 47, scope: !72)
