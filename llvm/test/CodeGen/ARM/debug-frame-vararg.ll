; RUN: llc -mtriple arm-unknown-linux-gnueabi -filetype asm -o - %s | FileCheck %s --check-prefix=CHECK-FP
; RUN: llc -mtriple arm-unknown-linux-gnueabi -filetype asm -o - %s -disable-fp-elim | FileCheck %s --check-prefix=CHECK-FP-ELIM
; RUN: llc -mtriple thumb-unknown-linux-gnueabi -filetype asm -o - %s | FileCheck %s --check-prefix=CHECK-THUMB-FP
; RUN: llc -mtriple thumb-unknown-linux-gnueabi -filetype asm -o - %s -disable-fp-elim | FileCheck %s --check-prefix=CHECK-THUMB-FP-ELIM

; Tests that the initial space allocated to the varargs on the stack is
; taken into account in the the .cfi_ directives.

; Generated from the C program:
; #include <stdarg.h>
;
; extern int foo(int);
;
; int sum(int count, ...) {
;  va_list vl;
;  va_start(vl, count);
;  int sum = 0;
;  for (int i = 0; i < count; i++) {
;   sum += foo(va_arg(vl, int));
;  }
;  va_end(vl);
; }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !{!"0x11\0012\00clang version 3.5 \000\00\000\00\000", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/var.c] [DW_LANG_C99]
!1 = !{!"var.c", !"/tmp"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00sum\00sum\00\005\000\001\000\006\00256\000\005", !1, !5, !6, null, i32 (i32, ...)* @sum, null, null, !2} ; [ DW_TAG_subprogram ] [line 5] [def] [sum]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/tmp/var.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 1, !"Debug Info Version", i32 2}
!11 = !{!"clang version 3.5 "}
!12 = !{!"0x101\00count\0016777221\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [count] [line 5]
!13 = !MDLocation(line: 5, scope: !4)
!14 = !{!"0x100\00vl\006\000", !4, !5, !15} ; [ DW_TAG_auto_variable ] [vl] [line 6]
!15 = !{!"0x16\00va_list\0030\000\000\000\000", !16, null, !17} ; [ DW_TAG_typedef ] [va_list] [line 30, size 0, align 0, offset 0] [from __builtin_va_list]
!16 = !{!"/linux-x86_64-high/gcc_4.7.2/dbg/llvm/bin/../lib/clang/3.5/include/stdarg.h", !"/tmp"}
!17 = !{!"0x16\00__builtin_va_list\006\000\000\000\000", !1, null, !18} ; [ DW_TAG_typedef ] [__builtin_va_list] [line 6, size 0, align 0, offset 0] [from __va_list]
!18 = !{!"0x13\00__va_list\006\0032\0032\000\000\000", !1, null, null, !19, null, null, null} ; [ DW_TAG_structure_type ] [__va_list] [line 6, size 32, align 32, offset 0] [def] [from ]
!19 = !{!20}
!20 = !{!"0xd\00__ap\006\0032\0032\000\000", !1, !18, !21} ; [ DW_TAG_member ] [__ap] [line 6, size 32, align 32, offset 0] [from ]
!21 = !{!"0xf\00\000\0032\0032\000\000", null, null, null} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from ]
!22 = !MDLocation(line: 6, scope: !4)
!23 = !MDLocation(line: 7, scope: !4)
!24 = !{!"0x100\00sum\008\000", !4, !5, !8} ; [ DW_TAG_auto_variable ] [sum] [line 8]
!25 = !MDLocation(line: 8, scope: !4)
!26 = !{!"0x100\00i\009\000", !27, !5, !8} ; [ DW_TAG_auto_variable ] [i] [line 9]
!27 = !{!"0xb\009\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [/tmp/var.c]
!28 = !MDLocation(line: 9, scope: !27)
!29 = !MDLocation(line: 10, scope: !30)
!30 = !{!"0xb\009\000\001", !1, !27} ; [ DW_TAG_lexical_block ] [/tmp/var.c]
!31 = !MDLocation(line: 11, scope: !30)
!32 = !MDLocation(line: 12, scope: !4)
!33 = !MDLocation(line: 13, scope: !4)

; CHECK-FP-LABEL: sum
; CHECK-FP: .cfi_startproc
; CHECK-FP: sub    sp, sp, #16
; CHECK-FP: .cfi_def_cfa_offset 16
; CHECK-FP: push   {r4, lr}
; CHECK-FP: .cfi_def_cfa_offset 24
; CHECK-FP: .cfi_offset lr, -20
; CHECK-FP: .cfi_offset r4, -24
; CHECK-FP: sub    sp, sp, #8
; CHECK-FP: .cfi_def_cfa_offset 32

; CHECK-FP-ELIM-LABEL: sum
; CHECK-FP-ELIM: .cfi_startproc
; CHECK-FP-ELIM: sub    sp, sp, #16
; CHECK-FP-ELIM: .cfi_def_cfa_offset 16
; CHECK-FP-ELIM: push   {r4, r10, r11, lr}
; CHECK-FP-ELIM: .cfi_def_cfa_offset 32
; CHECK-FP-ELIM: .cfi_offset lr, -20
; CHECK-FP-ELIM: .cfi_offset r11, -24
; CHECK-FP-ELIM: .cfi_offset r10, -28
; CHECK-FP-ELIM: .cfi_offset r4, -32
; CHECK-FP-ELIM: add    r11, sp, #8
; CHECK-FP-ELIM: .cfi_def_cfa r11, 24

; CHECK-THUMB-FP-LABEL: sum
; CHECK-THUMB-FP: .cfi_startproc
; CHECK-THUMB-FP: sub    sp, #16
; CHECK-THUMB-FP: .cfi_def_cfa_offset 16
; CHECK-THUMB-FP: push   {r4, r5, r7, lr}
; CHECK-THUMB-FP: .cfi_def_cfa_offset 32
; CHECK-THUMB-FP: .cfi_offset lr, -20
; CHECK-THUMB-FP: .cfi_offset r7, -24
; CHECK-THUMB-FP: .cfi_offset r5, -28
; CHECK-THUMB-FP: .cfi_offset r4, -32
; CHECK-THUMB-FP: sub    sp, #8
; CHECK-THUMB-FP: .cfi_def_cfa_offset 40

; CHECK-THUMB-FP-ELIM-LABEL: sum
; CHECK-THUMB-FP-ELIM: .cfi_startproc
; CHECK-THUMB-FP-ELIM: sub    sp, #16
; CHECK-THUMB-FP-ELIM: .cfi_def_cfa_offset 16
; CHECK-THUMB-FP-ELIM: push   {r4, r5, r7, lr}
; CHECK-THUMB-FP-ELIM: .cfi_def_cfa_offset 32
; CHECK-THUMB-FP-ELIM: .cfi_offset lr, -20
; CHECK-THUMB-FP-ELIM: .cfi_offset r7, -24
; CHECK-THUMB-FP-ELIM: .cfi_offset r5, -28
; CHECK-THUMB-FP-ELIM: .cfi_offset r4, -32
; CHECK-THUMB-FP-ELIM: add    r7, sp, #8
; CHECK-THUMB-FP-ELIM: .cfi_def_cfa r7, 24

define i32 @sum(i32 %count, ...) {
entry:
  %vl = alloca i8*, align 4
  %vl1 = bitcast i8** %vl to i8*
  call void @llvm.va_start(i8* %vl1)
  %cmp4 = icmp sgt i32 %count, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %ap.cur = load i8** %vl, align 4
  %ap.next = getelementptr i8* %ap.cur, i32 4
  store i8* %ap.next, i8** %vl, align 4
  %0 = bitcast i8* %ap.cur to i32*
  %1 = load i32* %0, align 4
  %call = call i32 @foo(i32 %1) #1
  %inc = add nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, %count
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  call void @llvm.va_end(i8* %vl1)
  ret i32 undef
}

declare void @llvm.va_start(i8*) nounwind

declare i32 @foo(i32)

declare void @llvm.va_end(i8*) nounwind
