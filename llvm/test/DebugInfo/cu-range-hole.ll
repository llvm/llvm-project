; REQUIRES: object-emission
; RUN: %llc_dwarf -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Check that we emit ranges for this CU since we have a function with and
; without debug info.
; Note: This depends upon the order of output in the .o file. Currently it's
; in order of the output to make sure that the CU has multiple ranges since
; there's a function in the middle. If they were together then it would have
; a single range and no DW_AT_ranges.
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_ranges
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram

; Function Attrs: nounwind uwtable
define i32 @b(i32 %c) #0 {
entry:
  %c.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %c.addr, metadata !13, metadata !{!"0x102"}), !dbg !14
  %0 = load i32* %c.addr, align 4, !dbg !14
  %add = add nsw i32 %0, 1, !dbg !14
  ret i32 %add, !dbg !14
}

; Function Attrs: nounwind uwtable
define i32 @a(i32 %b) #0 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32* %b.addr, align 4
  %add = add nsw i32 %0, 1
  ret i32 %add
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @d(i32 %e) #0 {
entry:
  %e.addr = alloca i32, align 4
  store i32 %e, i32* %e.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %e.addr, metadata !15, metadata !{!"0x102"}), !dbg !16
  %0 = load i32* %e.addr, align 4, !dbg !16
  %add = add nsw i32 %0, 1, !dbg !16
  ret i32 %add, !dbg !16
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0, !0}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!11, !12}

!0 = !{!"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)"}
!1 = !{!"0x11\0012\00clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)\000\00\000\00\001", !2, !3, !3, !4, !3, !3} ; [ DW_TAG_compile_unit ]
!2 = !{!"b.c", !"/usr/local/google/home/echristo"}
!3 = !{}
!4 = !{!5, !10}
!5 = !{!"0x2e\00b\00b\00\001\000\001\000\006\00256\000\001", !2, !6, !7, null, i32 (i32)* @b, null, null, !3} ; [ DW_TAG_subprogram ] [line 1] [def] [b]
!6 = !{!"0x29", !2}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/b.c]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9, !9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!"0x2e\00d\00d\00\003\000\001\000\006\00256\000\003", !2, !6, !7, null, i32 (i32)* @d, null, null, !3} ; [ DW_TAG_subprogram ] [line 3] [def] [d]
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 1, !"Debug Info Version", i32 2}
!13 = !{!"0x101\00c\0016777217\000", !5, !6, !9} ; [ DW_TAG_arg_variable ] [c] [line 1]
!14 = !MDLocation(line: 1, scope: !5)
!15 = !{!"0x101\00e\0016777219\000", !10, !6, !9} ; [ DW_TAG_arg_variable ] [e] [line 3]
!16 = !MDLocation(line: 3, scope: !10)
