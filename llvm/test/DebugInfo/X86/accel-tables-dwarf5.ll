; Verify the emission of accelerator tables for the DWARF v5 case.

; debug_names should be emitted regardless of the target and debugger tuning
; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj < %s \
; RUN:   | llvm-readobj -sections - | FileCheck --check-prefix=DEBUG_NAMES %s
; RUN: llc -mtriple=x86_64-apple-darwin12 -filetype=obj -debugger-tune=gdb < %s \
; RUN:   | llvm-readobj -sections - | FileCheck --check-prefix=DEBUG_NAMES %s
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj < %s \
; RUN:   | llvm-readobj -sections - | FileCheck --check-prefix=DEBUG_NAMES %s
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj -debugger-tune=lldb < %s \
; RUN:   | llvm-readobj -sections - | FileCheck --check-prefix=DEBUG_NAMES %s

; But not if also type units are enabled.
; TODO: This is the case because we currently don't generate DWARF v5-compatible
; type units. Change this once DWARF v5 type units are implemented.
; RUN: llc -mtriple=x86_64-pc-linux -filetype=obj -generate-type-units -debugger-tune=lldb < %s \
; RUN:   | llvm-readobj -sections - | FileCheck --check-prefix=NONE %s

; Debug types are ignored for non-ELF targets which means it shouldn't affect
; accelerator table generation.
; RUN: llc -mtriple=x86_64-apple-darwin12 -generate-type-units -filetype=obj < %s \
; RUN:   | llvm-readobj -sections - | FileCheck --check-prefix=DEBUG_NAMES %s

; NONE-NOT: apple_names
; NONE-NOT: debug_names

; DEBUG_NAMES-NOT: apple_names
; DEBUG_NAMES: debug_names
; DEBUG_NAMES-NOT: apple_names


@x = common dso_local global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !6, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 337260) (llvm/trunk 337262)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "-", directory: "/tmp", checksumkind: CSK_MD5, checksum: "06c25fe0c80b8959051a62f8f034710a")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "<stdin>", directory: "/tmp", checksumkind: CSK_MD5, checksum: "06c25fe0c80b8959051a62f8f034710a")
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 7.0.0 (trunk 337260) (llvm/trunk 337262)"}
