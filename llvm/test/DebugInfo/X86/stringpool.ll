; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s --check-prefix=LINUX
; RUN: llc -mtriple=x86_64-darwin < %s | FileCheck %s --check-prefix=DARWIN

source_filename = "test/DebugInfo/X86/stringpool.ll"

@yyyy = common global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "yyyy", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "z.c", directory: "/home/nicholas")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.1 (trunk 143009)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}

; Verify that we refer to 'yyyy' with a relocation.
; LINUX:      .long   .Linfo_string3          # DW_AT_name
; LINUX-NEXT: .long   {{[0-9]+}}              # DW_AT_type
; LINUX-NEXT:                                 # DW_AT_external
; LINUX-NEXT: .byte   1                       # DW_AT_decl_file
; LINUX-NEXT: .byte   1                       # DW_AT_decl_line
; LINUX-NEXT: .byte   9                       # DW_AT_location
; LINUX-NEXT: .byte   3
; LINUX-NEXT: .quad   yyyy
; Verify that "yyyy" ended up in the stringpool.
; LINUX: .section .debug_str,"MS",@progbits,1
; LINUX: yyyy

; Verify that we refer to 'yyyy' with a direct offset.
; DARWIN: .section        __DWARF,__debug_info,regular,debug
; DARWIN: DW_TAG_variable
; DARWIN:             .long   [[YYYY:[0-9]+]]
; DARWIN-NEXT:        .long   {{[0-9]+}}              ## DW_AT_type
; DARWIN-NEXT:                                        ## DW_AT_external
; DARWIN-NEXT:        .byte   1                       ## DW_AT_decl_file
; DARWIN-NEXT:        .byte   1                       ## DW_AT_decl_line
; DARWIN-NEXT:        .byte   9                       ## DW_AT_location
; DARWIN-NEXT:        .byte   3
; DARWIN-NEXT:        .quad   _yyyy
; DARWIN: .section __DWARF,__debug_str,regular,debug
; DARWIN: .asciz "yyyy" ## string offset=[[YYYY]]
!6 = !{!0}
!7 = !{i32 1, !"Debug Info Version", i32 3}
