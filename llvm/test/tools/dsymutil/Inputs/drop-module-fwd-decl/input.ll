; Test input for ../X86/drop-module-fwd-decl.test.
;
; A forward declaration of `X` nested under DW_TAG_module M, plus a real
; definition of `X` in another CU.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

; Two trivial functions so the debug map can list one symbol per CU.
define void @user_func() !dbg !100 { ret void }
define void @def_func() !dbg !200 {
  ; Local variable of type X so the X definition gets kept in the dSYM.
  %x.addr = alloca i32, align 4
    #dbg_declare(ptr %x.addr, !203, !DIExpression(), !204)
  store i32 42, ptr %x.addr, align 4, !dbg !204
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0, !30}
!llvm.module.flags = !{!50, !51}

; --- User CU: contains a DW_TAG_module M with a forward-declared struct X
;     inside. user_func references the module via DIImportedEntity.
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1,
                             producer: "test", emissionKind: FullDebug,
                             imports: !2, retainedTypes: !24)
!1 = !DIFile(filename: "user.cpp", directory: "/tmp")
!2 = !{!3}
!3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0,
                       entity: !22, line: 1)
!22 = !DIModule(scope: !0, name: "M", includePath: ".")
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "X", scope: !22,
                       flags: DIFlagFwdDecl, identifier: "_ZTS1X")
!24 = !{!23}
!100 = distinct !DISubprogram(name: "user_func", scope: !0, file: !1,
                              line: 1, type: !101, unit: !0,
                              spFlags: DISPFlagDefinition)
!101 = !DISubroutineType(types: !102)
!102 = !{null}

; --- Regular CU with the definition of struct X (scoped under another
;     DW_TAG_module M skeleton so it shares the candidate's synthetic
;     name) and def_func.
!30 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !31,
                              producer: "test", emissionKind: FullDebug,
                              imports: !37, retainedTypes: !36)
!31 = !DIFile(filename: "def.cpp", directory: "/tmp")
!37 = !{!38}
!38 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !30,
                        entity: !39, line: 1)
!39 = !DIModule(scope: !30, name: "M", includePath: ".")
!32 = !DICompositeType(tag: DW_TAG_structure_type, name: "X", scope: !39,
                       file: !31, line: 5, size: 32, elements: !33,
                       identifier: "_ZTS1X")
!33 = !{!34}
!34 = !DIDerivedType(tag: DW_TAG_member, name: "v", scope: !32, file: !31,
                     line: 6, baseType: !35, size: 32)
!35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!36 = !{!32}
!200 = distinct !DISubprogram(name: "def_func", scope: !30, file: !31,
                              line: 5, type: !101, unit: !30,
                              spFlags: DISPFlagDefinition,
                              retainedNodes: !210)
!203 = !DILocalVariable(name: "x", scope: !200, file: !31, line: 7, type: !32)
!204 = !DILocation(line: 7, column: 3, scope: !200)
!210 = !{!203}

!50 = !{i32 2, !"Dwarf Version", i32 4}
!51 = !{i32 2, !"Debug Info Version", i32 3}
