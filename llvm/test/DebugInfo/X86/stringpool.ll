; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s --check-prefix=LINUX
; RUN: llc -mtriple=x86_64-darwin < %s | FileCheck %s --check-prefix=DARWIN

@yyyy = common global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.1 (trunk 143009)", isOptimized: true, emissionKind: 0, file: !8, enums: !1, retainedTypes: !1, subprograms: !1, globals: !3, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = !DIGlobalVariable(name: "yyyy", line: 1, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7, variable: i32* @yyyy)
!6 = !DIFile(filename: "z.c", directory: "/home/nicholas")
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DIFile(filename: "z.c", directory: "/home/nicholas")

; Verify that "yyyy" ended up in the stringpool.
; LINUX: .section .debug_str,"MS",@progbits,1
; LINUX: yyyy
; DARWIN: .section __DWARF,__debug_str,regular,debug
; DARWIN: .asciz "yyyy" ## string offset=[[YYYY:[0-9]+]]

; Verify that we refer to 'yyyy' with a relocation.
; LINUX:      .long   .Linfo_string3          # DW_AT_name
; LINUX-NEXT: .long   {{[0-9]+}}              # DW_AT_type
; LINUX-NEXT:                                 # DW_AT_external
; LINUX-NEXT: .byte   1                       # DW_AT_decl_file
; LINUX-NEXT: .byte   1                       # DW_AT_decl_line
; LINUX-NEXT: .byte   9                       # DW_AT_location
; LINUX-NEXT: .byte   3
; LINUX-NEXT: .quad   yyyy

; Verify that we refer to 'yyyy' with a direct offset.
; DARWIN:             .long   [[YYYY]]
; DARWIN-NEXT:        .long   {{[0-9]+}}              ## DW_AT_type
; DARWIN-NEXT:                                        ## DW_AT_external
; DARWIN-NEXT:        .byte   1                       ## DW_AT_decl_file
; DARWIN-NEXT:        .byte   1                       ## DW_AT_decl_line
; DARWIN-NEXT:        .byte   9                       ## DW_AT_location
; DARWIN-NEXT:        .byte   3
; DARWIN-NEXT:        .quad   _yyyy
!9 = !{i32 1, !"Debug Info Version", i32 3}
