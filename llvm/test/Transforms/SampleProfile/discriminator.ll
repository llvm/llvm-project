; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/discriminator.prof | opt -analyze -branch-prob | FileCheck %s

; Original code
;
; 1   int foo(int i) {
; 2     int x = 0;
; 3     while (i < 100) {
; 4       if (i < 5) x--;
; 5       i++;
; 6     }
; 7     return x;
; 8   }
;
; In this test, if the loop is executed 100 times, the decrement operation
; at line 4 should only execute 5 times. This is reflected in the profile
; data for line offset 3.  In Inputs/discriminator.prof, we have:
;
; 3: 100
; 3.1: 5
;
; This means that the predicate 'i < 5' (line 3) is executed 100 times,
; but the then branch (line 3.1) is only executed 5 times.

define i32 @foo(i32 %i) #0 {
; CHECK: Printing analysis 'Branch Probability Analysis' for function 'foo':
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  store i32 0, i32* %x, align 4, !dbg !10
  br label %while.cond, !dbg !11

while.cond:                                       ; preds = %if.end, %entry
  %0 = load i32* %i.addr, align 4, !dbg !12
  %cmp = icmp slt i32 %0, 100, !dbg !12
  br i1 %cmp, label %while.body, label %while.end, !dbg !12
; CHECK: edge while.cond -> while.body probability is 100 / 101 = 99.0099% [HOT edge]
; CHECK: edge while.cond -> while.end probability is 1 / 101 = 0.990099%

while.body:                                       ; preds = %while.cond
  %1 = load i32* %i.addr, align 4, !dbg !14
  %cmp1 = icmp slt i32 %1, 50, !dbg !14
  br i1 %cmp1, label %if.then, label %if.end, !dbg !14
; CHECK: edge while.body -> if.then probability is 5 / 100 = 5%
; CHECK: edge while.body -> if.end probability is 95 / 100 = 95% [HOT edge]

if.then:                                          ; preds = %while.body
  %2 = load i32* %x, align 4, !dbg !17
  %dec = add nsw i32 %2, -1, !dbg !17
  store i32 %dec, i32* %x, align 4, !dbg !17
  br label %if.end, !dbg !17

if.end:                                           ; preds = %if.then, %while.body
  %3 = load i32* %i.addr, align 4, !dbg !19
  %inc = add nsw i32 %3, 1, !dbg !19
  store i32 %inc, i32* %i.addr, align 4, !dbg !19
  br label %while.cond, !dbg !20

while.end:                                        ; preds = %while.cond
  %4 = load i32* %x, align 4, !dbg !21
  ret i32 %4, !dbg !21
}


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{!"0x11\0012\00clang version 3.5 \000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [discriminator.c] [DW_LANG_C99]
!1 = !{!"discriminator.c", !"."}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, i32 (i32)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [discriminator.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.5 "}
!10 = !MDLocation(line: 2, scope: !4)
!11 = !MDLocation(line: 3, scope: !4)
!12 = !MDLocation(line: 3, scope: !13)
!13 = !{!"0xb\001", !1, !4} ; [ DW_TAG_lexical_block ] [discriminator.c]
!14 = !MDLocation(line: 4, scope: !15)
!15 = !{!"0xb\004\000\001", !1, !16} ; [ DW_TAG_lexical_block ] [discriminator.c]
!16 = !{!"0xb\003\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [discriminator.c]
!17 = !MDLocation(line: 4, scope: !18)
!18 = !{!"0xb\001", !1, !15} ; [ DW_TAG_lexical_block ] [discriminator.c]
!19 = !MDLocation(line: 5, scope: !16)
!20 = !MDLocation(line: 6, scope: !16)
!21 = !MDLocation(line: 7, scope: !4)
