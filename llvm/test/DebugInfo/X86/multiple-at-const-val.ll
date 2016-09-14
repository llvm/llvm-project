; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; rdar://13071590
; Check we are not emitting mutliple AT_const_value for a single member.
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_class_type
; CHECK: DW_TAG_member
; CHECK: badbit
; CHECK: DW_AT_const_value [DW_FORM_sdata]      (1)
; CHECK-NOT: DW_AT_const_value
; CHECK: NULL

%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_os" }
%"class.std::basic_os" = type { %"class.std::os_base", %"class.std::basic_ostream"*, i8, i8 }
%"class.std::os_base" = type { i32 (...)**, i64, i64, i32, i32, i32 }

@_ZSt4cout = external global %"class.std::basic_ostream"
@.str = private unnamed_addr constant [6 x i8] c"c is \00", align 1

define i32 @main() !dbg !960 {
entry:
  %call1.i = tail call %"class.std::basic_ostream"* @test(%"class.std::basic_ostream"* @_ZSt4cout, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), i64 5)
  ret i32 0
}

declare %"class.std::basic_ostream"* @test(%"class.std::basic_ostream"*, i8*, i64)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1803}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 174207)", isOptimized: true, emissionKind: FullDebug, file: !1802, enums: !1, retainedTypes: !955, globals: !1786, imports:  !955)
!1 = !{!26}
!4 = !DINamespace(name: "std", line: 48, scope: !5)
!5 = !DIFile(filename: "os_base.h", directory: "/privite/tmp")
!25 = !DIEnumerator(name: "_S_os_fmtflags_end", value: 65536) ; [ DW_TAG_enumerator ]
!26 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "_Ios_Iostate", line: 146, size: 32, align: 32, file: !1801, scope: !4, elements: !27)
!27 = !{!28, !29, !30, !31, !32}
!28 = !DIEnumerator(name: "_S_goodbit", value: 0) ; [ DW_TAG_enumerator ] [_S_goodbit :: 0]
!29 = !DIEnumerator(name: "_S_badbit", value: 1) ; [ DW_TAG_enumerator ] [_S_badbit :: 1]
!30 = !DIEnumerator(name: "_S_eofbit", value: 2) ; [ DW_TAG_enumerator ] [_S_eofbit :: 2]
!31 = !DIEnumerator(name: "_S_failbit", value: 4) ; [ DW_TAG_enumerator ] [_S_failbit :: 4]
!32 = !DIEnumerator(name: "_S_os_ostate_end", value: 65536) ; [ DW_TAG_enumerator ] [_S_os_ostate_end :: 65536]
!49 = !DICompositeType(tag: DW_TAG_class_type, name: "os_base", line: 200, size: 1728, align: 64, file: !1801, scope: !4, elements: !50, vtableHolder: !49)
!50 = !{!77}
!54 = !DISubroutineType(types: !55)
!55 = !{!56}
!56 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!77 = !DIDerivedType(tag: DW_TAG_member, name: "badbit", line: 331, flags: DIFlagStaticMember, file: !1801, scope: !49, baseType: !78, extraData: i32 1)
!78 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !79)
!79 = !DIDerivedType(tag: DW_TAG_typedef, name: "ostate", line: 327, file: !1801, scope: !49, baseType: !26)
!955 = !{}
!960 = distinct !DISubprogram(name: "main", line: 73, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 73, file: !1802, scope: null, type: !54, variables: !955)
!961 = !DIFile(filename: "student2.cpp", directory: "/privite/tmp")
!1786 = !{!1800}
!1800 = !DIGlobalVariable(name: "badbit", linkageName: "badbit", line: 331, isLocal: true, isDefinition: true, scope: !5, file: !5, type: !78, expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value), declaration: !77)
!1801 = !DIFile(filename: "os_base.h", directory: "/privite/tmp")
!1802 = !DIFile(filename: "student2.cpp", directory: "/privite/tmp")
!1803 = !{i32 1, !"Debug Info Version", i32 3}
