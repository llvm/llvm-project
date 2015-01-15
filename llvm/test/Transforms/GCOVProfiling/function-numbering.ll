; Test that GCOV instrumentation numbers functions correctly when some
; functions aren't emitted.

; Inject metadata to set the .gcno file location
; RUN: echo '!14 = !{!"%/T/function-numbering.ll", !0}' > %t1
; RUN: cat %s %t1 > %t2

; RUN: opt -insert-gcov-profiling -S < %t2 | FileCheck --check-prefix GCDA %s
; RUN: llvm-cov -n -dump %T/function-numbering.gcno 2>&1 | FileCheck --check-prefix GCNO %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; GCDA: @[[FOO:[0-9]+]] = private unnamed_addr constant [4 x i8] c"foo\00"
; GCDA-NOT: @{{[0-9]+}} = private unnamed_addr constant .* c"bar\00"
; GCDA: @[[BAZ:[0-9]+]] = private unnamed_addr constant [4 x i8] c"baz\00"
; GCDA: define internal void @__llvm_gcov_writeout()
; GCDA: call void @llvm_gcda_emit_function(i32 0, i8* getelementptr inbounds ([4 x i8]* @[[FOO]]
; GCDA: call void @llvm_gcda_emit_function(i32 1, i8* getelementptr inbounds ([4 x i8]* @[[BAZ]]

; GCNO: == foo (0) @
; GCNO-NOT: == bar ({{[0-9]+}}) @
; GCNO: == baz (1) @

define void @foo() {
  ret void, !dbg !12
}

define void @bar() {
  ; This function is referenced by the debug info, but no lines have locations.
  ret void
}

define void @baz() {
  ret void, !dbg !13
}

!llvm.gcov = !{!14}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !{!"0x11\0012\00clang version 3.6.0 \000\00\000\00\002", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [function-numbering.ll] [DW_LANG_C99]
!1 = !{!".../llvm/test/Transforms/GCOVProfiling/function-numbering.ll", !""}
!2 = !{}
!3 = !{!4, !7, !8}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\000\000\000\001", !1, !5, !6, null, void ()* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}    ; [ DW_TAG_file_type ] [/Users/bogner/build/llvm-debug//tmp/foo.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!"0x2e\00bar\00bar\00\002\000\001\000\000\000\000\002", !1, !5, !6, null, void ()* @bar, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [def] [bar]
!8 = !{!"0x2e\00baz\00baz\00\003\000\001\000\000\000\000\003", !1, !5, !6, null, void ()* @baz, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [baz]
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 2}
!11 = !{!"clang version 3.6.0 "}
!12 = !MDLocation(line: 1, column: 13, scope: !4)
!13 = !MDLocation(line: 3, column: 13, scope: !8)
