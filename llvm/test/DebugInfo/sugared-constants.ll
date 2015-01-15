; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj %s -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Use correct signedness when emitting constants of derived (sugared) types.

; Test compiled to IR from clang with -O1 and the following source:

; void func(int);
; void func(unsigned);
; void func(char16_t);
; int main() {
;   const int i = 42;
;   func(i);
;   const unsigned j = 117;
;   func(j);
;   char16_t c = 7;
;   func(c);
; }

; CHECK: DW_AT_const_value [DW_FORM_sdata] (42)
; CHECK: DW_AT_const_value [DW_FORM_udata] (117)
; CHECK: DW_AT_const_value [DW_FORM_udata] (7)

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  tail call void @llvm.dbg.value(metadata i32 42, i64 0, metadata !10, metadata !{!"0x102"}), !dbg !21
  tail call void @_Z4funci(i32 42), !dbg !22
  tail call void @llvm.dbg.value(metadata i32 117, i64 0, metadata !12, metadata !{!"0x102"}), !dbg !24
  tail call void @_Z4funcj(i32 117), !dbg !25
  tail call void @llvm.dbg.value(metadata i16 7, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !27
  tail call void @_Z4funcDs(i16 zeroext 7), !dbg !28
  ret i32 0, !dbg !29
}

declare void @_Z4funci(i32) #1

declare void @_Z4funcj(i32) #1

declare void @_Z4funcDs(i16 zeroext) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = !{!"0x11\004\00clang version 3.5.0 \001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/const.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"const.cpp", !"/tmp/dbginfo"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00main\00main\00\004\000\001\000\006\00256\001\004", !1, !5, !6, null, i32 ()* @main, null, null, !9} ; [ DW_TAG_subprogram ] [line 4] [def] [main]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/const.cpp]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!10, !12, !15}
!10 = !{!"0x100\00i\005\000", !4, !5, !11} ; [ DW_TAG_auto_variable ] [i] [line 5]
!11 = !{!"0x26\00\000\000\000\000\000", null, null, !8} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from int]
!12 = !{!"0x100\00j\007\000", !4, !5, !13} ; [ DW_TAG_auto_variable ] [j] [line 7]
!13 = !{!"0x26\00\000\000\000\000\000", null, null, !14} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from unsigned int]
!14 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!15 = !{!"0x100\00c\009\000", !4, !5, !16} ; [ DW_TAG_auto_variable ] [c] [line 9]
!16 = !{!"0x24\00char16_t\000\0016\0016\000\000\0016", null, null} ; [ DW_TAG_base_type ] [char16_t] [line 0, size 16, align 16, offset 0, enc DW_ATE_UTF]
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 1, !"Debug Info Version", i32 2}
!19 = !{!"clang version 3.5.0 "}
!20 = !{i32 42}
!21 = !MDLocation(line: 5, scope: !4)
!22 = !MDLocation(line: 6, scope: !4)
!23 = !{i32 117}
!24 = !MDLocation(line: 7, scope: !4)
!25 = !MDLocation(line: 8, scope: !4)
!26 = !{i16 7}
!27 = !MDLocation(line: 9, scope: !4)
!28 = !MDLocation(line: 10, scope: !4)
!29 = !MDLocation(line: 11, scope: !4)
