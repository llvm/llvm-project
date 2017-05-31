; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

@e = global i16 0, align 2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.2 (trunk 165274) (llvm/trunk 165272)", isOptimized: false, emissionKind: FullDebug, file: !8, enums: !1, retainedTypes: !1, globals: !3, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = !DIGlobalVariable(name: "e", line: 2, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7, variable: i16* @e)
!6 = !DIFile(filename: "foo.cpp", directory: "/tmp")
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", line: 1, size: 16, align: 16, flags: DIFlagFwdDecl, file: !8)
!8 = !DIFile(filename: "foo.cpp", directory: "/tmp")

; CHECK: DW_TAG_enumeration_type
; CHECK-NEXT: DW_AT_name
; CHECK-NEXT: DW_AT_byte_size
; CHECK-NEXT: DW_AT_declaration
!9 = !{i32 1, !"Debug Info Version", i32 3}
