source_filename = "DW_AT_GNU_template_name.cpp"
target triple = "x86_64-pc-linux-gnu"

%class.Baz = type { %class.Bar }
%class.Bar = type { %class.Foo }
%class.Foo = type { i32 }

@TT = dso_local global %class.Baz zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!22, !23}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "TT", scope: !2, file: !3, line: 18, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "DW_AT_GNU_template_name.cpp", directory: "")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "Example", file: !3, line: 16, baseType: !6)
!6 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Baz<Bar>", file: !3, line: 12, size: 32, flags: DIFlagTypePassByValue, elements: !7, templateParams: !20, identifier: "_ZTS3BazI3BarE")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "Foo", scope: !6, file: !3, line: 13, baseType: !9, size: 32)
!9 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Bar<Foo>", file: !3, line: 7, size: 32, flags: DIFlagTypePassByValue, elements: !10, templateParams: !18, identifier: "_ZTS3BarI3FooE")
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "Int", scope: !9, file: !3, line: 8, baseType: !12, size: 32)
!12 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Foo<int>", file: !3, line: 3, size: 32, flags: DIFlagTypePassByValue, elements: !13, templateParams: !16, identifier: "_ZTS3FooIiE")
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "Member", scope: !12, file: !3, line: 3, baseType: !15, size: 32)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17}
!17 = !DITemplateTypeParameter(name: "T", type: !15)
!18 = !{!19}
!19 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "TemplateType", value: !"Foo")
!20 = !{!21}
!21 = !DITemplateValueParameter(tag: DW_TAG_GNU_template_template_param, name: "TemplateTemplateType", value: !"Bar")
!22 = !{i32 7, !"Dwarf Version", i32 5}
!23 = !{i32 2, !"Debug Info Version", i32 3}
