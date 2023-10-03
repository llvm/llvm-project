; RUN: llvm-as -disable-output %s 2>&1 | FileCheck --match-full-lines %s

!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!0}
!0 = distinct !DICompileUnit(language: 0, file: !1)
!1 = !DIFile(filename: "-", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: invalid file
define void @invalid_file() !dbg !3 { ret void }
!3 = distinct !DISubprogram(file: !0)

; CHECK: line specified with no file
define void @line_specified_with_no_file() !dbg !4 { ret void }
!4 = distinct !DISubprogram(line: 1)

; CHECK: invalid subroutine type
define void @invalid_subroutine_type() !dbg !5 { ret void }
!5 = distinct !DISubprogram(type: !0)

; CHECK: invalid containing type
define void @invalid_containing_type() !dbg !6 { ret void }
!6 = distinct !DISubprogram(containingType: !0)

; CHECK: invalid template params
define void @invalid_template_params() !dbg !7 { ret void }
!7 = distinct !DISubprogram(templateParams: !0)

; CHECK: invalid template parameter
define void @invalid_template_parameter() !dbg !8 { ret void }
!8 = distinct !DISubprogram(templateParams: !{!0})

; CHECK: invalid subprogram declaration
define void @invalid_subprogram_declaration() !dbg !9 { ret void }
!9 = distinct !DISubprogram(declaration: !0)

; CHECK: invalid retained nodes list
define void @invalid_retained_nodes_list() !dbg !10 { ret void }
!10 = distinct !DISubprogram(retainedNodes: !0)

; CHECK: invalid retained nodes, expected DILocalVariable, DILabel or DIImportedEntity
define void @invalid_retained_nodes_expected() !dbg !11 { ret void }
!11 = distinct !DISubprogram(retainedNodes: !{!0})

; CHECK: invalid reference flags
define void @invalid_reference_flags_reference() !dbg !12 { ret void }
!12 = distinct !DISubprogram(flags: DIFlagLValueReference | DIFlagRValueReference)

; CHECK: invalid reference flags
define void @invalid_reference_flags_pass_by() !dbg !13 { ret void }
!13 = distinct !DISubprogram(flags: DIFlagTypePassByValue | DIFlagTypePassByReference)

; CHECK: subprogram definitions must have a compile unit
define void @subprogram_definitions_must_have_a_compile_unit() !dbg !14 { ret void }
!14 = distinct !DISubprogram()

; CHECK: invalid unit type
define void @invalid_unit_type() !dbg !15 { ret void }
!15 = distinct !DISubprogram(unit: !{})

; FIXME: should something verify `isDefinition` is not a lie? is it meaningful
; to mistmatch it with respect to the LLVM IR function?
; CHECK: subprogram declarations must not have a compile unit
define void @subprogram_declarations_must_not_have_a_compile_unit() !dbg !16 { ret void }
!16 = distinct !DISubprogram(isDefinition: false, unit: !0)

; CHECK: invalid thrown types list
define void @invalid_thrown_types_list() !dbg !17 { ret void }
!17 = distinct !DISubprogram(isDefinition: false, thrownTypes: !0)

; CHECK: invalid thrown type
define void @invalid_thrown_type() !dbg !18 { ret void }
!18 = distinct !DISubprogram(isDefinition: false, thrownTypes: !{!0})

; CHECK: DIFlagAllCallsDescribed must be attached to a definition
define void @DIFlagAllCallsDescribed_must_be_attached_to_a_definition() !dbg !19 { ret void }
!19 = distinct !DISubprogram(isDefinition: false, flags: DIFlagAllCallsDescribed)

; CHECK: warning: ignoring invalid debug info{{.*}}
