; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Make sure that structures have a decl file and decl line attached.
; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_decl_file
; CHECK: DW_AT_decl_line
; CHECK: DW_TAG_member

%struct.foo = type { i32 }

@f = common global %struct.foo zeroinitializer, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.1 (trunk 152837) (llvm/trunk 152845)", isOptimized: false, emissionKind: FullDebug, file: !11, enums: !1, retainedTypes: !1, globals: !3, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = !DIGlobalVariable(name: "f", line: 5, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7, variable: %struct.foo* @f)
!6 = !DIFile(filename: "struct_bug.c", directory: "/Users/echristo/tmp")
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 1, size: 32, align: 32, file: !11, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !11, scope: !7, baseType: !10)
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DIFile(filename: "struct_bug.c", directory: "/Users/echristo/tmp")
!12 = !{i32 1, !"Debug Info Version", i32 3}
