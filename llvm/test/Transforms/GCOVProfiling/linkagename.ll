; RUN: echo '!9 = !{!"%/T/linkagename.ll", !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output < %t2
; RUN: grep _Z3foov %T/linkagename.gcno
; RUN: rm %T/linkagename.gcno

define void @_Z3foov() {
entry:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}
!llvm.gcov = !{!9}

!0 = !{!"0x11\004\00clang version 3.3 (trunk 177323)\000\00\000\00\000", !2, !3, !3, !4, !3,  !3} ; [ DW_TAG_compile_unit ] [/home/nlewycky/hello.cc] [DW_LANG_C_plus_plus]
!1 = !{!"0x29", !2}          ; [ DW_TAG_file_type ] [/home/nlewycky/hello.cc]
!2 = !{!"hello.cc", !"/home/nlewycky"}
!3 = !{i32 0}
!4 = !{!5}
!5 = !{!"0x2e\00foo\00foo\00_Z3foov\001\000\001\000\006\00256\000\001", !1, !1, !6, null, void ()* @_Z3foov, null, null, !3} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !MDLocation(line: 1, scope: !5)


!10 = !{i32 1, !"Debug Info Version", i32 2}
