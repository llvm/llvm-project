target triple = "x86_64-apple-macosx"
; RUN: %llc_dwarf %s -o - -filetype=obj \
; RUN:   | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; CHECK: DW_TAG_module
; CHECK-NOT: NULL
; CHECK: DW_TAG_structure_type

; Hand-crafted based on
; struct s;
; struct s *s;

%struct.s = type opaque

@i = common global %struct.s* null, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, globals: !3, imports: !11)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "s", scope: !0, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, variable: %struct.s** @i)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64)
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "s", scope: !9, file: !1, line: 1, flags: DIFlagFwdDecl)
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DIModule(scope: null, name: "Module", configMacros: "", includePath: ".", isysroot: "/")
!10 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !9, line: 11)
!11 = !{!10}
