; RUN: echo '!9 = !{!"%/T/version.ll", !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output < %t2
; RUN: head -c8 %T/version.gcno | grep '^oncg.204'
; RUN: rm %T/version.gcno
; RUN: not opt -insert-gcov-profiling -default-gcov-version=asdfasdf -disable-output < %t2
; RUN: opt -insert-gcov-profiling -default-gcov-version=407* -disable-output < %t2
; RUN: head -c8 %T/version.gcno | grep '^oncg.704'
; RUN: rm %T/version.gcno

define void @test() {
  ret void, !dbg !8
}

!llvm.gcov = !{!9}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = !{!"0x11\004\00clang version 3.3 (trunk 176994)\000\00\000\00\000", !11, !3, !3, !4, !3, null} ; [ DW_TAG_compile_unit ] [./version] [DW_LANG_C_plus_plus]
!2 = !{!"0x29", !11} ; [ DW_TAG_file_type ]
!3 = !{i32 0}
!4 = !{!5}
!5 = !{!"0x2e\00test\00test\00\001\000\001\000\006\00256\000\001", !10, !6, !7, null, void ()* @test, null, null, !3} ; [ DW_TAG_subprogram ] [line 1] [def] [test]
!6 = !{!"0x29", !10} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !3, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !MDLocation(line: 1, scope: !5)
;; !9 is added through the echo line at the top.
!10 = !{!"<stdin>", !"."}
!11 = !{!"version", !"/usr/local/google/home/nlewycky"}
!12 = !{i32 1, !"Debug Info Version", i32 2}
