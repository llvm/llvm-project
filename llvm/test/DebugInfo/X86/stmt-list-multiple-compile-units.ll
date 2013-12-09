; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; RUN: llc < %s -O0 -mtriple=x86_64-apple-macosx10.7 | FileCheck %s -check-prefix=ASM

; rdar://13067005
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
; CHECK: DW_AT_stmt_list [DW_FORM_data4]   (0x00000000)

; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
; CHECK: DW_AT_stmt_list [DW_FORM_data4]   (0x0000003c)

; CHECK: .debug_line contents:
; CHECK-NEXT: Line table prologue:
; CHECK-NEXT: total_length: 0x00000038
; CHECK: file_names[  1]    0 0x00000000 0x00000000 simple.c
; CHECK: Line table prologue:
; CHECK-NEXT: total_length: 0x00000039
; CHECK: file_names[  1]    0 0x00000000 0x00000000 simple2.c
; CHECK-NOT: file_names

; PR15408
; ASM: L__DWARF__debug_info_begin0:
; ASM: .long   0                       ## DW_AT_stmt_list
; ASM: L__DWARF__debug_info_begin1:
; ASM: .long   0                       ## DW_AT_stmt_list
define i32 @test(i32 %a) nounwind uwtable ssp {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !15), !dbg !16
  %0 = load i32* %a.addr, align 4, !dbg !17
  %call = call i32 @fn(i32 %0), !dbg !17
  ret i32 %call, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @fn(i32 %a) nounwind uwtable ssp {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !19), !dbg !20
  %0 = load i32* %a.addr, align 4, !dbg !21
  ret i32 %0, !dbg !21
}

!llvm.dbg.cu = !{!0, !10}
!llvm.module.flags = !{!25}
!0 = metadata !{i32 786449, metadata !23, i32 12, metadata !"clang version 3.3", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !23, metadata !6, metadata !"test", metadata !"test", metadata !"", i32 2, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @test, null, null, metadata !1, i32 3} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 3] [test]
!6 = metadata !{i32 786473, metadata !23} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786449, metadata !24, i32 12, metadata !"clang version 3.3 (trunk 172862)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !11, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!11 = metadata !{metadata !13}
!13 = metadata !{i32 786478, metadata !24, metadata !14, metadata !"fn", metadata !"fn", metadata !"", i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @fn, null, null, metadata !1, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [fn]
!14 = metadata !{i32 786473, metadata !24} ; [ DW_TAG_file_type ]
!15 = metadata !{i32 786689, metadata !5, metadata !"a", metadata !6, i32 16777218, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 2]
!16 = metadata !{i32 2, i32 0, metadata !5, null}
!17 = metadata !{i32 4, i32 0, metadata !18, null}
!18 = metadata !{i32 786443, metadata !23, metadata !5, i32 3, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 786689, metadata !13, metadata !"a", metadata !14, i32 16777217, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 1]
!20 = metadata !{i32 1, i32 0, metadata !13, null}
!21 = metadata !{i32 2, i32 0, metadata !22, null}
!22 = metadata !{i32 786443, metadata !24, metadata !13, i32 1, i32 0, i32 0} ; [ DW_TAG_lexical_block ]
!23 = metadata !{metadata !"simple.c", metadata !"/private/tmp"}
!24 = metadata !{metadata !"simple2.c", metadata !"/private/tmp"}
!25 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
