; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Make sure that structures have a decl file and decl line attached.
; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_decl_file
; CHECK: DW_AT_decl_line
; CHECK: DW_TAG_member

source_filename = "test/DebugInfo/X86/struct-loc.ll"

%struct.foo = type { i32 }

@f = common global %struct.foo zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "f", scope: null, file: !2, line: 5, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "struct_bug.c", directory: "/Users/echristo/tmp")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !2, line: 1, size: 32, align: 32, elements: !4)
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !2, line: 2, baseType: !6, size: 32, align: 32)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.1 (trunk 152837) (llvm/trunk 152845)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, retainedTypes: !8, globals: !9, imports: !8)
!8 = !{}
!9 = !{!0}
!10 = !{i32 1, !"Debug Info Version", i32 3}

