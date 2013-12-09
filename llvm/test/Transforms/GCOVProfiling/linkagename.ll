; RUN: echo '!9 = metadata !{metadata !"%T/linkagename.ll", metadata !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output < %t2
; RUN: grep _Z3foov %T/linkagename.gcno
; RUN: rm %T/linkagename.gcno

; REQUIRES: shell

define void @_Z3foov() {
entry:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}
!llvm.gcov = !{!9}

!0 = metadata !{i32 786449, metadata !2, i32 4, metadata !"clang version 3.3 (trunk 177323)", i1 false, metadata !"", i32 0, metadata !3, metadata !3, metadata !4, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ] [/home/nlewycky/hello.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 786473, metadata !2}          ; [ DW_TAG_file_type ] [/home/nlewycky/hello.cc]
!2 = metadata !{metadata !"hello.cc", metadata !"/home/nlewycky"}
!3 = metadata !{i32 0}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !1, metadata !1, metadata !"foo", metadata !"foo", metadata !"_Z3foov", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z3foov, null, null, metadata !3, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!6 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 1, i32 0, metadata !5, null}


!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
