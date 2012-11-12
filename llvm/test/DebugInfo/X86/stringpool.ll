; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s --check-prefix=LINUX
; RUN: llc -mtriple=x86_64-darwin < %s | FileCheck %s --check-prefix=DARWIN

@yyyy = common global i32 0, align 4

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 12, metadata !"z.c", metadata !"/home/nicholas", metadata !"clang version 3.1 (trunk 143009)", i1 true, i1 true, metadata !"", i32 0, metadata !1, metadata !1, metadata !1, metadata !3} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720948, i32 0, null, metadata !"yyyy", metadata !"yyyy", metadata !"", metadata !6, i32 1, metadata !7, i32 0, i32 1, i32* @yyyy} ; [ DW_TAG_variable ]
!6 = metadata !{i32 720937, metadata !"z.c", metadata !"/home/nicholas", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]

; Verify that we refer to 'yyyy' with a relocation.
; LINUX:      .long   .Lstring3               # DW_AT_name
; LINUX-NEXT: .long   38                      # DW_AT_type
; LINUX-NEXT:                                 # DW_AT_external
; LINUX-NEXT: .byte   1                       # DW_AT_decl_file
; LINUX-NEXT: .byte   1                       # DW_AT_decl_line
; LINUX-NEXT: .byte   9                       # DW_AT_location
; LINUX-NEXT: .byte   3
; LINUX-NEXT: .quad   yyyy

; Verify that we refer to 'yyyy' without a relocation.
; DARWIN: Lset5 = Lstring3-Lsection_str               ## DW_AT_name
; DARWIN-NEXT:        .long   Lset5
; DARWIN-NEXT:        .long   39                      ## DW_AT_type
; DARWIN-NEXT:        .byte   1                       ## DW_AT_external
; DARWIN-NEXT:        .byte   1                       ## DW_AT_decl_file
; DARWIN-NEXT:        .byte   1                       ## DW_AT_decl_line
; DARWIN-NEXT:        .byte   9                       ## DW_AT_location
; DARWIN-NEXT:        .byte   3
; DARWIN-NEXT:        .quad   _yyyy

; Verify that "yyyy" ended up in the stringpool.
; LINUX: .section .debug_str,"MS",@progbits,1
; LINUX-NOT: .section
; LINUX: yyyy
; DARWIN: .section __DWARF,__debug_str,regular,debug
; DARWIN-NOT: .section
; DARWIN: yyyy
