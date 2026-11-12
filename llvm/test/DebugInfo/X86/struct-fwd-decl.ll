; RUN: llc -O0 -mtriple=x86_64-unknown-linux %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; Test that size is not emitted for class declarations in DWARF, even if it exists.

@s = global i16 0, align 2, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "s", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "foo.cpp", directory: "/tmp")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !2, line: 1, size: 16, align: 16, flags: DIFlagFwdDecl)
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name
; CHECK-NOT: DW_AT_byte_size
; CHECK: {{NULL|DW_TAG}}
!6 = !{!0}
!7 = !{i32 1, !"Debug Info Version", i32 3}
