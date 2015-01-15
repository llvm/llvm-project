; RUN: llc -mtriple=i386-unknown-unknown -mcpu=core2 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=line %t | FileCheck %s
;
; Generated from:
;
;   int foo(int i) {
;     if (i < 10) return i - 1;
;     return 0;
;   }
;
; Manually generated debug nodes !14 and !15 to incorporate an
; arbitrary discriminator with value 42.

define i32 @foo(i32 %i) #0 {
entry:
  %retval = alloca i32, align 4
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.end, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32* %i.addr, align 4, !dbg !14
  %sub = sub nsw i32 %1, 1, !dbg !14
  store i32 %sub, i32* %retval, !dbg !14
  br label %return, !dbg !14

if.end:                                           ; preds = %entry
  store i32 0, i32* %retval, !dbg !12
  br label %return, !dbg !12

return:                                           ; preds = %if.end, %if.then
  %2 = load i32* %retval, !dbg !13
  ret i32 %2, !dbg !13
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{!"0x11\0012\00clang version 3.5 \000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [./discriminator.c] [DW_LANG_C99]
!1 = !{!"discriminator.c", !"."}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, i32 (i32)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [./discriminator.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.5 "}
!10 = !MDLocation(line: 2, scope: !11)
!11 = !{!"0xb\002\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [./discriminator.c]
!12 = !MDLocation(line: 3, scope: !4)
!13 = !MDLocation(line: 4, scope: !4)
!14 = !MDLocation(line: 2, scope: !15)
!15 = !{!"0xb\0042", !1, !4} ; [ DW_TAG_lexical_block ] [./discriminator.c]

; CHECK: Address            Line   Column File   ISA Discriminator Flags
; CHECK: ------------------ ------ ------ ------ --- ------------- -------------
; CHECK: 0x0000000000000011      2      0      1   0            42 {{$}}
