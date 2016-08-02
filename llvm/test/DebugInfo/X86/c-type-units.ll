; REQUIRES: object-emission

; RUN: llc -o - %s -filetype=obj -O0 -generate-dwarf-pub-sections=Disable -generate-type-units -mtriple=x86_64-unknown-linux-gnu | llvm-dwarfdump -debug-dump=types - | FileCheck %s

; struct foo {
; } f;

; no known LLVM frontends produce appropriate unique identifiers for C types,
; so we don't produce type units for them
; CHECK-NOT: DW_TAG_type_unit

%struct.foo = type {}

@f = common global %struct.foo zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "simple.c", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "f", line: 2, isLocal: false, isDefinition: true, scope: null, file: !5, type: !6, variable: %struct.foo* @f)
!5 = !DIFile(filename: "simple.c", directory: "/tmp/dbginfo")
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 1, align: 8, file: !1, elements: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5 "}
