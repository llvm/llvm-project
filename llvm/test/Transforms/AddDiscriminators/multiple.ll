; RUN: opt < %s -add-discriminators -S | FileCheck %s

; Discriminator support for multiple CFG paths on the same line.
;
;       void foo(int i) {
;         int x;
;         if (i < 10) x = i; else x = -i;
;       }
;
; The two stores inside the if-then-else line must have different discriminator
; values.

define void @foo(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.else, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32* %i.addr, align 4, !dbg !10
; CHECK:  %1 = load i32* %i.addr, align 4, !dbg !12

  store i32 %1, i32* %x, align 4, !dbg !10
; CHECK:  store i32 %1, i32* %x, align 4, !dbg !12

  br label %if.end, !dbg !10
; CHECK:  br label %if.end, !dbg !12

if.else:                                          ; preds = %entry
  %2 = load i32* %i.addr, align 4, !dbg !10
; CHECK:  %2 = load i32* %i.addr, align 4, !dbg !14

  %sub = sub nsw i32 0, %2, !dbg !10
; CHECK:  %sub = sub nsw i32 0, %2, !dbg !14

  store i32 %sub, i32* %x, align 4, !dbg !10
; CHECK:  store i32 %sub, i32* %x, align 4, !dbg !14

  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !12
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !{!"0x11\0012\00clang version 3.5 (trunk 199750) (llvm/trunk 199751)\000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [multiple.c] [DW_LANG_C99]
!1 = !{!"multiple.c", !"."}
!2 = !{i32 0}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", !1, !5, !6, null, void (i32)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [multiple.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 2}
!9 = !{!"clang version 3.5 (trunk 199750) (llvm/trunk 199751)"}
!10 = !MDLocation(line: 3, scope: !11)
!11 = !{!"0xb\003\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [multiple.c]
!12 = !MDLocation(line: 4, scope: !4)

; CHECK: !12 = !MDLocation(line: 3, scope: !13)
; CHECK: !13 = !{!"0xb\001", !1, !11} ; [ DW_TAG_lexical_block ] [./multiple.c]
; CHECK: !14 = !MDLocation(line: 3, scope: !15)
; CHECK: !15 = !{!"0xb\002", !1, !11} ; [ DW_TAG_lexical_block ] [./multiple.c]
