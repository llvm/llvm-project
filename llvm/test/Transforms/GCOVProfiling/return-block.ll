; Inject metadata to set the .gcno file location
; RUN: echo '!19 = !{!"%/T/return-block.ll", !0}' > %t1
; RUN: cat %s %t1 > %t2

; By default, the return block is last.
; RUN: opt -insert-gcov-profiling -disable-output %t2
; RUN: llvm-cov gcov -n -dump %T/return-block.gcno 2>&1 | FileCheck -check-prefix=CHECK -check-prefix=RETURN-LAST %s

; But we can optionally emit it second, to match newer gcc versions.
; RUN: opt -insert-gcov-profiling -gcov-exit-block-before-body -disable-output %t2
; RUN: llvm-cov gcov -n -dump %T/return-block.gcno 2>&1 | FileCheck -check-prefix=CHECK -check-prefix=RETURN-SECOND %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @test() #0 {
entry:
  tail call void (...)* @f() #2, !dbg !14
  %0 = load i32* @A, align 4, !dbg !15
  %tobool = icmp eq i32 %0, 0, !dbg !15
  br i1 %tobool, label %if.end, label %if.then, !dbg !15

if.then:                                          ; preds = %entry
  tail call void (...)* @g() #2, !dbg !16
  br label %if.end, !dbg !16

if.end:                                           ; preds = %entry, %if.then
  ret void, !dbg !18
}

declare void @f(...) #1

declare void @g(...) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.gcov = !{!19}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = !{!"0x11\0012\00clang version 3.6.0 (trunk 223182)\001\00\000\00\001", !1, !2, !2, !3, !8, !2} ; [ DW_TAG_compile_unit ] [return-block.c] [DW_LANG_C99]
!1 = !{!".../llvm/test/Transforms/GCOVProfiling/return-block.ll", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00test\00test\00\005\000\001\000\000\000\001\005", !1, !5, !6, null, void ()* @test, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [test]
!5 = !{!"0x29", !1}    ; [ DW_TAG_file_type ] [return-block.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!9}
!9 = !{!"0x34\00A\00A\00\003\000\001", null, !5, !10, i32* @A, null} ; [ DW_TAG_variable ] [A] [line 3] [def]
!10 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 2}
!13 = !{!"clang version 3.6.0 (trunk 223182)"}
!14 = !MDLocation(line: 6, column: 3, scope: !4)
!15 = !MDLocation(line: 7, column: 7, scope: !4)
!16 = !MDLocation(line: 8, column: 5, scope: !17)
!17 = !{!"0xb\007\007\000", !1, !4} ; [ DW_TAG_lexical_block ] [return-block.c]
!18 = !MDLocation(line: 9, column: 1, scope: !4)

; There should be no destination edges for the exit block.
; CHECK: Block : 1 Counter : 0
; RETURN-LAST:       Destination Edges
; RETURN-SECOND-NOT: Destination Edges
; CHECK: Block : 2 Counter : 0
; CHECK: Block : 4 Counter : 0
; RETURN-LAST-NOT: Destination Edges
; RETURN-SECOND:   Destination Edges
; CHECK-NOT: Block :
