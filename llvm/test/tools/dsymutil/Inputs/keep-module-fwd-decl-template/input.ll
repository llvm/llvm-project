; Test input for ../X86/keep-module-fwd-decl-template.test.
;
; A templated forward declaration `Foo<T>` nested under DW_TAG_module M.
; No definition for Foo exists anywhere, so the forward decl is kept and
; must retain its DW_TAG_template_type_parameter child.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

define void @user_func() !dbg !100 { ret void }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!50, !51}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1,
                             producer: "test", emissionKind: FullDebug,
                             imports: !2, retainedTypes: !24)
!1 = !DIFile(filename: "user.cpp", directory: "/tmp")
!2 = !{!3}
!3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0,
                       entity: !22, line: 1)
!22 = !DIModule(scope: !0, name: "M", includePath: ".")
!23 = !DICompositeType(tag: DW_TAG_class_type, name: "Foo<T>", scope: !22,
                       flags: DIFlagFwdDecl, identifier: "_ZTS3FooIT_E",
                       templateParams: !26)
!24 = !{!23}
!25 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!26 = !{!27}
!27 = !DITemplateTypeParameter(name: "T", type: !25)
!100 = distinct !DISubprogram(name: "user_func", scope: !0, file: !1,
                              line: 1, type: !101, unit: !0,
                              spFlags: DISPFlagDefinition)
!101 = !DISubroutineType(types: !102)
!102 = !{null}

!50 = !{i32 2, !"Dwarf Version", i32 4}
!51 = !{i32 2, !"Debug Info Version", i32 3}
