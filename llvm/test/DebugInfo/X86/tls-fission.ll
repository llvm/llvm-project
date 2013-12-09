; RUN: llc -split-dwarf=Enable -mtriple=x86_64-linux -O0 -filetype=asm < %s | FileCheck %s

; FIXME: add relocation and DWARF expression support to llvm-dwarfdump & use
; that here instead of raw assembly printing

; CHECK: debug_info.dwo
; 3 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; CHECK: .byte 3{{ *}}# DW_AT_location
; DW_OP_const_index (0xfx == 252) to refer to the debug_addr table
; CHECK-NEXT: .byte 252
; an index of zero into the debug_addr table
; CHECK-NEXT: .byte 0
; DW_OP_lo_user based on GCC/GDB extension presumably (by experiment) to support TLS
; CHECK-NEXT: .byte 224
; check that the expected TLS address description is the first thing in the debug_addr section
; CHECK: debug_addr
; CHECK-NEXT: .quad	tls@DTPOFF

@tls = thread_local global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !3, metadata !2, metadata !"tls.dwo"} ; [ DW_TAG_compile_unit ] [/tmp/tls.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"tls.cpp", metadata !"/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786484, i32 0, null, metadata !"tls", metadata !"tls", metadata !"", metadata !5, i32 1, metadata !6, i32 0, i32 1, i32* @tls, null} ; [ DW_TAG_variable ] [tls] [line 1] [def]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/tls.cpp]
!6 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
