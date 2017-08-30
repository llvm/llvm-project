; REQUIRES: object-emission

; RUN: llc -o - %s -filetype=obj -O0 -generate-dwarf-pub-sections=Disable -generate-type-units -mtriple=x86_64-unknown-linux-gnu | llvm-dwarfdump -debug-dump=types - | FileCheck %s

; struct foo {
; } f;

; no known LLVM frontends produce appropriate unique identifiers for C types,
; so we don't produce type units for them
; CHECK-NOT: DW_TAG_type_unit

source_filename = "test/DebugInfo/X86/c-type-units.ll"

%struct.foo = type {}

@f = common global %struct.foo zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "f", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "simple.c", directory: "/tmp/dbginfo")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !2, line: 1, align: 8, elements: !4)
!4 = !{}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.5 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !6, imports: !4)
!6 = !{!0}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5 "}

