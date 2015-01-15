; REQUIRES: object-emission
; RUN: %llc_dwarf -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Check that we emit ranges for this which has a non-traditional section and a normal section.

; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_ranges
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_low_pc
; CHECK: DW_AT_high_pc
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_low_pc
; CHECK: DW_AT_high_pc

; CHECK: .debug_ranges contents:
; FIXME: When we get better dumping facilities we'll want to elaborate here.
; CHECK: 00000000 <End of list>

; Function Attrs: nounwind uwtable
define i32 @foo(i32 %a) #0 section "__TEXT,__foo" {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !13, metadata !{!"0x102"}), !dbg !14
  %0 = load i32* %a.addr, align 4, !dbg !15
  %add = add nsw i32 %0, 5, !dbg !15
  ret i32 %add, !dbg !15
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !16, metadata !{!"0x102"}), !dbg !17
  %0 = load i32* %a.addr, align 4, !dbg !18
  %add = add nsw i32 %0, 5, !dbg !18
  ret i32 %add, !dbg !18
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = !{!"0x11\0012\00clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/foo.c] [DW_LANG_C99]
!1 = !{!"foo.c", !"/usr/local/google/home/echristo"}
!2 = !{}
!3 = !{!4, !9}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, i32 (i32)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/foo.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x2e\00bar\00bar\00\005\000\001\000\006\00256\000\005", !1, !5, !6, null, i32 (i32)* @bar, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [bar]
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 1, !"Debug Info Version", i32 2}
!12 = !{!"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)"}
!13 = !{!"0x101\00a\0016777217\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [a] [line 1]
!14 = !MDLocation(line: 1, scope: !4)
!15 = !MDLocation(line: 2, scope: !4)
!16 = !{!"0x101\00a\0016777221\000", !9, !5, !8} ; [ DW_TAG_arg_variable ] [a] [line 5]
!17 = !MDLocation(line: 5, scope: !9)
!18 = !MDLocation(line: 6, scope: !9)

