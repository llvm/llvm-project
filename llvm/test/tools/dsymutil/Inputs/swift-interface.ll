; This is a manually stripped empty Swift program with one import.
source_filename = "/swift-interface.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

@__swift_reflection_version = linkonce_odr hidden constant i16 3
@llvm.used = appending global [1 x i8*] [i8* bitcast (i16* @__swift_reflection_version to i8*)], section "llvm.metadata", align 8

define i32 @main(i32, i8**) !dbg !29 {
entry:
  %2 = bitcast i8** %1 to i8*
  ret i32 0, !dbg !35
}

!llvm.dbg.cu = !{!0}
!swift.module.flags = !{!14}
!llvm.module.flags = !{!20, !21, !24}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, enums: !2, imports: !3)
!1 = !DIFile(filename: "ParseableInterfaceImports.swift", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !1, entity: !5, file: !1)
!5 = !DIModule(scope: null, name: "Foo", includePath: "/Foo/x86_64.swiftinterface")
!14 = !{!"standard-library", i1 false}
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{i32 1, !"Swift Version", i32 7}
!29 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !5, file: !1, line: 1, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{}
!35 = !DILocation(line: 0, scope: !36)
!36 = !DILexicalBlockFile(scope: !29, file: !37, discriminator: 0)
!37 = !DIFile(filename: "<compiler-generated>", directory: "")
