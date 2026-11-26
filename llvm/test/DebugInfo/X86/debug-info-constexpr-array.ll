; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj %s -o %t
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s

; Test that DW_AT_const_value is emitted for constexpr arrays of integer types
; (char, int, uint16_t, uint32_t, uint64_t, negative ints) and NOT emitted
; for 128-bit integers or floating-point arrays.
;
; Reduced from clang output for:
;   using uint16_t = unsigned short;
;   using uint32_t = unsigned int;
;   using uint64_t = unsigned long long;
;   struct Test {
;     static inline constexpr char STR[] = "Hello";
;     static inline constexpr int NUMS[] = {1, 2, 3};
;     static inline constexpr uint16_t SHORTS[] = {256, 512, 1024};
;     static inline constexpr uint32_t INTS[] = {70000, 80000};
;     static inline constexpr uint64_t LONGS[] = {4294967296ULL, 123456789ULL};
;     static inline constexpr int NEG[] = {-1, -128, 42};
;     static inline constexpr __int128_t I128[] = {1, -1};
;     static inline constexpr __uint128_t U128[] = {1, 2};
;     static inline constexpr float FLOATS[] = {1.0f, 2.0f};
;     static inline constexpr double DOUBLES[] = {1.0, 2.0};
;   };
;
; CHECK: DW_TAG_structure_type
; CHECK:   DW_AT_name ("Test")

; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("STR")
; CHECK:     DW_AT_const_value (<0x06> 48 65 6c 6c 6f 00 )

; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("NUMS")
; CHECK:     DW_AT_const_value (<0x0c> 01 00 00 00 02 00 00 00 03 00 00 00 )

; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("SHORTS")
; CHECK:     DW_AT_const_value (<0x06> 00 01 00 02 00 04 )

; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("INTS")
; CHECK:     DW_AT_const_value (<0x08> 70 11 01 00 80 38 01 00 )

; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("LONGS")
; CHECK:     DW_AT_const_value (<0x10> 00 00 00 00 01 00 00 00 15 cd 5b 07 00 00 00 00 )

; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("NEG")
; CHECK:     DW_AT_const_value (<0x0c> ff ff ff ff 80 ff ff ff 2a 00 00 00 )

; 128-bit integers: no DW_AT_const_value.
; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("I128")
; CHECK-NOT: DW_AT_const_value
; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("U128")
; CHECK-NOT: DW_AT_const_value

; Floating-point arrays: no DW_AT_const_value.
; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("FLOATS")
; CHECK-NOT: DW_AT_const_value

; CHECK:   DW_TAG_member
; CHECK:     DW_AT_name ("DOUBLES")
; CHECK-NOT: DW_AT_const_value

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZN4Test3STRE = linkonce_odr constant [6 x i8] c"Hello\00", align 1, !dbg !0
@_ZN4Test4NUMSE = linkonce_odr constant [3 x i32] [i32 1, i32 2, i32 3], align 4, !dbg !5
@_ZN4Test6SHORTSE = linkonce_odr constant [3 x i16] [i16 256, i16 512, i16 1024], align 2, !dbg !58
@_ZN4Test4INTSE = linkonce_odr constant [2 x i32] [i32 70000, i32 80000], align 4, !dbg !60
@_ZN4Test5LONGSE = linkonce_odr constant [2 x i64] [i64 4294967296, i64 123456789], align 16, !dbg !62
@_ZN4Test3NEGE = linkonce_odr constant [3 x i32] [i32 -1, i32 -128, i32 42], align 4, !dbg !64
@_ZN4Test4I128E = linkonce_odr constant [2 x i128] [i128 1, i128 -1], align 16, !dbg !66
@_ZN4Test4U128E = linkonce_odr constant [2 x i128] [i128 1, i128 2], align 16, !dbg !68
@_ZN4Test6FLOATSE = linkonce_odr constant [2 x float] [float 1.000000e+00, float 2.000000e+00], align 4, !dbg !70
@_ZN4Test7DOUBLESE = linkonce_odr constant [2 x double] [double 1.000000e+00, double 2.000000e+00], align 16, !dbg !72

define dso_local void @_Z4testv() !dbg !77 {
entry:
  call void @_Z3usePKv(ptr @_ZN4Test3STRE), !dbg !80
  call void @_Z3usePKv(ptr @_ZN4Test4NUMSE), !dbg !81
  call void @_Z3usePKv(ptr @_ZN4Test6SHORTSE), !dbg !82
  call void @_Z3usePKv(ptr @_ZN4Test4INTSE), !dbg !83
  call void @_Z3usePKv(ptr @_ZN4Test5LONGSE), !dbg !84
  call void @_Z3usePKv(ptr @_ZN4Test3NEGE), !dbg !85
  call void @_Z3usePKv(ptr @_ZN4Test4I128E), !dbg !86
  call void @_Z3usePKv(ptr @_ZN4Test4U128E), !dbg !87
  call void @_Z3usePKv(ptr @_ZN4Test6FLOATSE), !dbg !88
  call void @_Z3usePKv(ptr @_ZN4Test7DOUBLESE), !dbg !89
  ret void, !dbg !90
}

declare void @_Z3usePKv(ptr)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!74, !75}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "STR", linkageName: "_ZN4Test3STRE", scope: !2, file: !7, line: 9, type: !17, isLocal: false, isDefinition: true, declaration: !16)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/tmp")
!4 = !{!0, !5, !58, !60, !62, !64, !66, !68, !70, !72}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "NUMS", linkageName: "_ZN4Test4NUMSE", scope: !2, file: !7, line: 10, type: !8, isLocal: false, isDefinition: true, declaration: !13)
!7 = !DIFile(filename: "test.cpp", directory: "/tmp")
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 96, elements: !11)
!9 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 3)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "NUMS", scope: !14, file: !7, line: 10, baseType: !8, flags: DIFlagStaticMember, extraData: [3 x i32] [i32 1, i32 2, i32 3])
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Test", file: !7, line: 8, size: 8, flags: DIFlagTypePassByValue, elements: !15, identifier: "_ZTS4Test")
!15 = !{!16, !13, !22, !27, !34, !39, !40, !45, !50, !54}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "STR", scope: !14, file: !7, line: 9, baseType: !17, flags: DIFlagStaticMember, extraData: [6 x i8] c"Hello\00")
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 48, elements: !20)
!18 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !19)
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!20 = !{!21}
!21 = !DISubrange(count: 6)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "SHORTS", scope: !14, file: !7, line: 11, baseType: !23, flags: DIFlagStaticMember, extraData: [3 x i16] [i16 256, i16 512, i16 1024])
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 48, elements: !11)
!24 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !25)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint16_t", file: !7, line: 4, baseType: !26)
!26 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "INTS", scope: !14, file: !7, line: 12, baseType: !28, flags: DIFlagStaticMember, extraData: [2 x i32] [i32 70000, i32 80000])
!28 = !DICompositeType(tag: DW_TAG_array_type, baseType: !29, size: 64, elements: !32)
!29 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !30)
!30 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !7, line: 5, baseType: !31)
!31 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!32 = !{!33}
!33 = !DISubrange(count: 2)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "LONGS", scope: !14, file: !7, line: 13, baseType: !35, flags: DIFlagStaticMember, extraData: [2 x i64] [i64 4294967296, i64 123456789])
!35 = !DICompositeType(tag: DW_TAG_array_type, baseType: !36, size: 128, elements: !32)
!36 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !37)
!37 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !7, line: 6, baseType: !38)
!38 = !DIBasicType(name: "unsigned long long", size: 64, encoding: DW_ATE_unsigned)
!39 = !DIDerivedType(tag: DW_TAG_member, name: "NEG", scope: !14, file: !7, line: 14, baseType: !8, flags: DIFlagStaticMember, extraData: [3 x i32] [i32 -1, i32 -128, i32 42])
!40 = !DIDerivedType(tag: DW_TAG_member, name: "I128", scope: !14, file: !7, line: 17, baseType: !41, flags: DIFlagStaticMember)
!41 = !DICompositeType(tag: DW_TAG_array_type, baseType: !42, size: 256, elements: !32)
!42 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !43)
!43 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int128_t", file: !3, baseType: !44)
!44 = !DIBasicType(name: "__int128", size: 128, encoding: DW_ATE_signed)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "U128", scope: !14, file: !7, line: 18, baseType: !46, flags: DIFlagStaticMember)
!46 = !DICompositeType(tag: DW_TAG_array_type, baseType: !47, size: 256, elements: !32)
!47 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !48)
!48 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint128_t", file: !3, baseType: !49)
!49 = !DIBasicType(name: "unsigned __int128", size: 128, encoding: DW_ATE_unsigned)
!50 = !DIDerivedType(tag: DW_TAG_member, name: "FLOATS", scope: !14, file: !7, line: 19, baseType: !51, flags: DIFlagStaticMember)
!51 = !DICompositeType(tag: DW_TAG_array_type, baseType: !52, size: 64, elements: !32)
!52 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !53)
!53 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!54 = !DIDerivedType(tag: DW_TAG_member, name: "DOUBLES", scope: !14, file: !7, line: 20, baseType: !55, flags: DIFlagStaticMember)
!55 = !DICompositeType(tag: DW_TAG_array_type, baseType: !56, size: 128, elements: !32)
!56 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !57)
!57 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!58 = !DIGlobalVariableExpression(var: !59, expr: !DIExpression())
!59 = distinct !DIGlobalVariable(name: "SHORTS", linkageName: "_ZN4Test6SHORTSE", scope: !2, file: !7, line: 11, type: !23, isLocal: false, isDefinition: true, declaration: !22)
!60 = !DIGlobalVariableExpression(var: !61, expr: !DIExpression())
!61 = distinct !DIGlobalVariable(name: "INTS", linkageName: "_ZN4Test4INTSE", scope: !2, file: !7, line: 12, type: !28, isLocal: false, isDefinition: true, declaration: !27)
!62 = !DIGlobalVariableExpression(var: !63, expr: !DIExpression())
!63 = distinct !DIGlobalVariable(name: "LONGS", linkageName: "_ZN4Test5LONGSE", scope: !2, file: !7, line: 13, type: !35, isLocal: false, isDefinition: true, declaration: !34)
!64 = !DIGlobalVariableExpression(var: !65, expr: !DIExpression())
!65 = distinct !DIGlobalVariable(name: "NEG", linkageName: "_ZN4Test3NEGE", scope: !2, file: !7, line: 14, type: !8, isLocal: false, isDefinition: true, declaration: !39)
!66 = !DIGlobalVariableExpression(var: !67, expr: !DIExpression())
!67 = distinct !DIGlobalVariable(name: "I128", linkageName: "_ZN4Test4I128E", scope: !2, file: !7, line: 17, type: !41, isLocal: false, isDefinition: true, declaration: !40)
!68 = !DIGlobalVariableExpression(var: !69, expr: !DIExpression())
!69 = distinct !DIGlobalVariable(name: "U128", linkageName: "_ZN4Test4U128E", scope: !2, file: !7, line: 18, type: !46, isLocal: false, isDefinition: true, declaration: !45)
!70 = !DIGlobalVariableExpression(var: !71, expr: !DIExpression())
!71 = distinct !DIGlobalVariable(name: "FLOATS", linkageName: "_ZN4Test6FLOATSE", scope: !2, file: !7, line: 19, type: !51, isLocal: false, isDefinition: true, declaration: !50)
!72 = !DIGlobalVariableExpression(var: !73, expr: !DIExpression())
!73 = distinct !DIGlobalVariable(name: "DOUBLES", linkageName: "_ZN4Test7DOUBLESE", scope: !2, file: !7, line: 20, type: !55, isLocal: false, isDefinition: true, declaration: !54)
!74 = !{i32 2, !"Debug Info Version", i32 3}
!75 = !{i32 1, !"wchar_size", i32 4}
!77 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", scope: !7, file: !7, line: 24, type: !78, scopeLine: 24, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!78 = !DISubroutineType(types: !79)
!79 = !{null}
!80 = !DILocation(line: 25, column: 3, scope: !77)
!81 = !DILocation(line: 26, column: 3, scope: !77)
!82 = !DILocation(line: 27, column: 3, scope: !77)
!83 = !DILocation(line: 28, column: 3, scope: !77)
!84 = !DILocation(line: 29, column: 3, scope: !77)
!85 = !DILocation(line: 30, column: 3, scope: !77)
!86 = !DILocation(line: 31, column: 3, scope: !77)
!87 = !DILocation(line: 32, column: 3, scope: !77)
!88 = !DILocation(line: 33, column: 3, scope: !77)
!89 = !DILocation(line: 34, column: 3, scope: !77)
!90 = !DILocation(line: 35, column: 1, scope: !77)
