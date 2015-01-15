; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t -dwarf-version=3
; RUN: llvm-dwarfdump %t | FileCheck %s -check-prefix=DWARF3
; RUN: llc < %s -O0 -mtriple=x86_64-apple-macosx10.7 | FileCheck %s -check-prefix=ASM

; rdar://13067005
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_stmt_list [DW_FORM_sec_offset]   (0x00000000)
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000000)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000010)
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000000)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000010)

; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_stmt_list [DW_FORM_sec_offset]   (0x0000003c)
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000010)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000009)
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000010)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000009)


; CHECK: .debug_line contents:
; CHECK-NEXT: Line table prologue:
; CHECK-NEXT: total_length: 0x00000038
; CHECK: file_names[  1]    0 0x00000000 0x00000000 simple.c
; CHECK: Line table prologue:
; CHECK-NEXT: total_length: 0x00000039
; CHECK: file_names[  1]    0 0x00000000 0x00000000 simple2.c
; CHECK-NOT: file_names

; DWARF3: .debug_info contents:
; DWARF3: DW_TAG_compile_unit
; DWARF3: DW_AT_stmt_list [DW_FORM_data4]    (0x00000000)

; DWARF3: DW_TAG_compile_unit
; DWARF3: DW_AT_stmt_list [DW_FORM_data4]   (0x0000003c)


; DWARF3: .debug_line contents:
; DWARF3-NEXT: Line table prologue:
; DWARF3-NEXT: total_length: 0x00000038
; DWARF3: file_names[  1]    0 0x00000000 0x00000000 simple.c
; DWARF3: Line table prologue:
; DWARF3-NEXT: total_length: 0x00000039
; DWARF3: file_names[  1]    0 0x00000000 0x00000000 simple2.c
; DWARF3-NOT: file_names

; PR15408
; ASM: L__DWARF__debug_info_begin0:
; ASM: Lset3 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
; ASM-NEXT: .long   Lset3
; ASM: L__DWARF__debug_info_begin1:
; ASM: Lset13 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
; ASM-NEXT: .long   Lset13
define i32 @test(i32 %a) nounwind uwtable ssp {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !15, metadata !{!"0x102"}), !dbg !16
  %0 = load i32* %a.addr, align 4, !dbg !17
  %call = call i32 @fn(i32 %0), !dbg !17
  ret i32 %call, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @fn(i32 %a) nounwind uwtable ssp {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !19, metadata !{!"0x102"}), !dbg !20
  %0 = load i32* %a.addr, align 4, !dbg !21
  ret i32 %0, !dbg !21
}

!llvm.dbg.cu = !{!0, !10}
!llvm.module.flags = !{!25}
!0 = !{!"0x11\0012\00clang version 3.3\000\00\000\00\001", !23, !1, !1, !3, !1,  !1} ; [ DW_TAG_compile_unit ]
!1 = !{}
!3 = !{!5}
!5 = !{!"0x2e\00test\00test\00\002\000\001\000\006\00256\000\003", !23, !6, !7, null, i32 (i32)* @test, null, null, !1} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 3] [test]
!6 = !{!"0x29", !23} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x11\0012\00clang version 3.3 (trunk 172862)\000\00\000\00\001", !24, !1, !1, !11, !1,  !1} ; [ DW_TAG_compile_unit ]
!11 = !{!13}
!13 = !{!"0x2e\00fn\00fn\00\001\000\001\000\006\00256\000\001", !24, !14, !7, null, i32 (i32)* @fn, null, null, !1} ; [ DW_TAG_subprogram ] [line 1] [def] [fn]
!14 = !{!"0x29", !24} ; [ DW_TAG_file_type ]
!15 = !{!"0x101\00a\0016777218\000", !5, !6, !9} ; [ DW_TAG_arg_variable ] [a] [line 2]
!16 = !MDLocation(line: 2, scope: !5)
!17 = !MDLocation(line: 4, scope: !18)
!18 = !{!"0xb\003\000\000", !23, !5} ; [ DW_TAG_lexical_block ]
!19 = !{!"0x101\00a\0016777217\000", !13, !14, !9} ; [ DW_TAG_arg_variable ] [a] [line 1]
!20 = !MDLocation(line: 1, scope: !13)
!21 = !MDLocation(line: 2, scope: !22)
!22 = !{!"0xb\001\000\000", !24, !13} ; [ DW_TAG_lexical_block ]
!23 = !{!"simple.c", !"/private/tmp"}
!24 = !{!"simple2.c", !"/private/tmp"}
!25 = !{i32 1, !"Debug Info Version", i32 2}
