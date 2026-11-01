; RUN: llvm-as -disable-output %s 2>&1 | FileCheck --match-full-lines %s

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!2, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !17, !20, !22}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "-", directory: "")

; CHECK: invalid file
!2 = distinct !DICompileUnit(language: 0, file: !{})

; CHECK: invalid filename
!3 = !DIFile(filename: "", directory: "")
!4 = distinct !DICompileUnit(language: 0, file: !3)

; CHECK: invalid enum list
!5 = distinct !DICompileUnit(language: 0, file: !1, enums: !1)

; CHECK: invalid enum type
!6 = distinct !DICompileUnit(language: 0, file: !1, enums: !{!1})

; CHECK: invalid retained type list
!7 = distinct !DICompileUnit(language: 0, file: !1, retainedTypes: !1)

; CHECK: invalid retained type
!8 = distinct !DICompileUnit(language: 0, file: !1, retainedTypes: !{!1})

; CHECK: invalid global variable list
!9 = distinct !DICompileUnit(language: 0, file: !1, globals: !1)

; CHECK: invalid global variable ref
!10 = distinct !DICompileUnit(language: 0, file: !1, globals: !{!1})

; CHECK: invalid imported entity list
!11 = distinct !DICompileUnit(language: 0, file: !1, imports: !1)

; CHECK: invalid imported entity ref
!12 = distinct !DICompileUnit(language: 0, file: !1, imports: !{!1})

; CHECK: invalid macro list
!13 = distinct !DICompileUnit(language: 0, file: !1, macros: !1)

; CHECK: invalid macro ref
!14 = distinct !DICompileUnit(language: 0, file: !1, macros: !{!1})

define void @foo() !dbg !15 { ret void }
!15 = distinct !DISubprogram(file: !1, type: !23)

; CHECK: function-local imports are not allowed in a DICompileUnit's imported entities list
!16 = distinct !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !15, entity: !15)
!17 = distinct !DICompileUnit(language: 0, file: !1, imports: !{!16})

; CHECK: function-local variables are not allowed in a DICompileUnit's global variables list
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = distinct !DIGlobalVariable(scope: !15)
!20 = distinct !DICompileUnit(language: 0, file: !1, globals: !{!18})

; CHECK: function-local enum in a DICompileUnit's enum list
!21 = distinct !DICompositeType(tag: DW_TAG_enumeration_type, scope: !15)
!22 = distinct !DICompileUnit(language: 0, file: !1, enums: !{!21})

; CHECK: warning: ignoring invalid debug info{{.*}}

!23 = !DISubroutineType(types: !24)
!24 = !{null}
