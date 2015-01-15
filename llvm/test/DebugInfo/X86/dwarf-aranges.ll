; RUN: llc -generate-arange-section < %s | FileCheck %s


; -- header --
; CHECK: .short 2 # DWARF Arange version number
; CHECK-NEXT: .long .L.debug_info_begin0
; CHECK-NEXT: .byte 8 # Address Size (in bytes)
; CHECK-NEXT: .byte 0 # Segment Size (in bytes)
; -- alignment --
; CHECK-NEXT: .zero 4,255

; <common symbols> - it should have made one span for each symbol.
; CHECK-NEXT: .quad some_bss
; CHECK-NEXT: .quad 4

; <data section> - it should have made one span covering all vars in this CU.
; CHECK-NEXT: .quad some_data
; CHECK-NEXT: .quad .Ldebug_end1-some_data

; <text section> - it should have made one span covering all functions in this CU.
; CHECK-NEXT: .quad .Lfunc_begin0
; CHECK-NEXT: .quad .Ldebug_end2-.Lfunc_begin0

; <other sections> - it should have made one span covering all vars in this CU.
; CHECK-NEXT: .quad some_other
; CHECK-NEXT: .quad .Ldebug_end3-some_other

; -- finish --
; CHECK-NEXT: # ARange terminator



; -- source code --
; Generated from: "clang -c -g -emit-llvm"
;
; int some_data = 4;
; int some_bss;
; int some_other __attribute__ ((section ("strange+section"))) = 5;
; 
; void some_code()
; {
;    some_bss += some_data + some_other;
; }

target triple = "x86_64-unknown-linux-gnu"

@some_data = global i32 4, align 4
@some_other = global i32 5, section "strange+section", align 4
@some_bss = common global i32 0, align 4

define void @some_code() {
entry:
  %0 = load i32* @some_data, align 4, !dbg !14
  %1 = load i32* @some_other, align 4, !dbg !14
  %add = add nsw i32 %0, %1, !dbg !14
  %2 = load i32* @some_bss, align 4, !dbg !14
  %add1 = add nsw i32 %2, %add, !dbg !14
  store i32 %add1, i32* @some_bss, align 4, !dbg !14
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !16}

!0 = !{!"0x11\0012\00clang version 3.4 \000\00\000\00\000", !1, !2, !2, !3, !8, !2} ; [ DW_TAG_compile_unit ] [/home/kayamon/test.c] [DW_LANG_C99]
!1 = !{!"test.c", !"/home/kayamon"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00some_code\00some_code\00\005\000\001\000\006\000\000\006", !1, !5, !6, null, void ()* @some_code, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [scope 6] [some_code]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/home/kayamon/test.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!9, !11, !12}
!9 = !{!"0x34\00some_data\00some_data\00\001\000\001", null, !5, !10, i32* @some_data, null} ; [ DW_TAG_variable ] [some_data] [line 1] [def]
!10 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = !{!"0x34\00some_other\00some_other\00\003\000\001", null, !5, !10, i32* @some_other, null} ; [ DW_TAG_variable ] [some_other] [line 3] [def]
!12 = !{!"0x34\00some_bss\00some_bss\00\002\000\001", null, !5, !10, i32* @some_bss, null} ; [ DW_TAG_variable ] [some_bss] [line 2] [def]
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !MDLocation(line: 7, scope: !4)
!15 = !MDLocation(line: 8, scope: !4)
!16 = !{i32 1, !"Debug Info Version", i32 2}
