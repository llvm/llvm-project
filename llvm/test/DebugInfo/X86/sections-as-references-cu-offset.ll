; Test that references to CU are 4 bytes long when the target is 64-bit
; and -dwarf-sections-as-references=Enable is specified.

; RUN: llc -filetype=asm -mtriple=x86_64-linux-gnu -dwarf-sections-as-references=Enable < %s \
; RUN:   | FileCheck %s

; CHECK: .section .debug_pubnames
; CHECK: .long .debug_info      # Offset of Compilation Unit Info
; CHECK: .section .debug_pubtypes
; CHECK: .long .debug_info      # Offset of Compilation Unit Info

%struct.foo = type { i8 }

@f = global %struct.foo zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "f", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "tu1.cpp", directory: "/dir")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !2, line: 1, size: 8, align: 8, elements: !4, identifier: "_ZTS3foo")
!4 = !{}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !6, globals: !7, imports: !4)
!6 = !{!3}
!7 = !{!0}
!8 = !{i32 1, !"Debug Info Version", i32 3}
