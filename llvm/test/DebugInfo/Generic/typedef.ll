; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; From source:
; typedef void x;
; x *y;

; Check that a typedef with no DW_AT_type is produced. The absence of a type is used to imply the 'void' type.

; CHECK: DW_TAG_typedef
; CHECK-NOT: DW_AT_type
; CHECK: {{DW_TAG|NULL}}

@y = global i8* null, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "typedef.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "y", line: 2, isLocal: false, isDefinition: true, scope: null, file: !5, type: !6, variable: i8** @y)
!5 = !DIFile(filename: "typedef.cpp", directory: "/tmp/dbginfo")
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "x", line: 1, file: !1, baseType: null)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5.0 "}

