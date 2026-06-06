;; Check if basic blocks without unique sections are only placed in cold sections if it is safe
;; to do so.
;;
;; Profile for version 0.
; RUN: echo 'v1' > %t1
; RUN: echo 'f _Z3asm_goto' >> %t1
; RUN: echo 'c 0' >> %t1
; RUN: echo 'f _Z3jump_table' >> %t1
; RUN: echo 'c 0' >> %t1
; RUN: echo 'f _Z3red_zone' >> %t1
; RUN: echo 'c 0' >> %t1
;;
; RUN: llc < %s -mtriple=aarch64 -function-sections -basic-block-sections=%t1 -unique-basic-block-section-names -bbsections-cold-text-prefix=".text.unlikely." | FileCheck %s
; RUN: llc < %s -mtriple=aarch64 -function-sections -aarch64-min-jump-table-entries=4 -basic-block-sections=%t1 -unique-basic-block-section-names -bbsections-cold-text-prefix=".text.unlikely." | FileCheck %s -check-prefix=JUMP-TABLES
; RUN: llc < %s -mtriple=aarch64 -function-sections -basic-block-sections=%t1 -unique-basic-block-section-names -bbsections-cold-text-prefix=".text.unlikely." | FileCheck %s -check-prefix=RED-ZONE

define void @_Z3asm_goto(i1 zeroext %0, i1 zeroext %1) nounwind {
  ;; Check that blocks containing or targeted by asm goto aren't split.
  ; CHECK-LABEL:  _Z3asm_goto
  ; CHECK:        .section	.text.unlikely._Z3asm_goto,"ax",@progbits
  ; CHECK-NEXT:     _Z3asm_goto.cold:
  ; CHECK-NEXT:       bl bam
  ; CHECK:          .LBB0_4:
  ; CHECK:            ret
  ; CHECK:          .LBB_END0_4:

  br i1 %0, label %3, label %5

3:                                                ; preds = %2
  %4 = call i32 @bar()
  callbr void asm sideeffect "nop", "!i"() #3
          to label %asm.fallthrough [label %5]


asm.fallthrough:                                  ; preds = %3
    br label %5

5:                                                ; preds = %2, %asm.fallthrough
  %6 = call i32 @bar()
  br i1 %1, label %7, label %9

7:
  %8 = call i32 @bam()
  br label %9

9:                                                ; preds = %7
  ret void
}

define i32 @_Z3jump_table(i32 %in) nounwind {
  ;; Check that a cold block that contains a jump table dispatch or
  ;; that is targeted by a jump table is not split.
  ; JUMP-TABLES-LABEL:  _Z3jump_table
  ; JUMP-TABLES:        .section	.text.unlikely._Z3jump_table,"ax",@progbits
  ; JUMP-TABLES-NEXT:     _Z3jump_table.cold:
  ; JUMP-TABLES-SAME:                         %common.ret
  ; JUMP-TABLES-NOT:        b       bar
  ; JUMP-TABLES-NOT:        b       baz
  ; JUMP-TABLES-NOT:        b       qux
  ; JUMP-TABLES-NOT:        b       bam

  switch i32 %in, label %common.ret [
    i32 0, label %cold1
    i32 1, label %cold2
    i32 2, label %cold3
    i32 3, label %cold4
  ]

  common.ret:                                       ; preds = %0
    ret i32 0

  cold1:                                            ; preds = %0
    %1 = tail call i32 @bar()
    ret i32 %1

  cold2:                                            ; preds = %0
    %2 = tail call i32 @baz()
    ret i32 %2

  cold3:                                            ; preds = %0
    %3 = tail call i32 @bam()
    ret i32 %3

  cold4:                                            ; preds = %0
    %4 = tail call i32 @qux()
    ret i32 %4
}

define i32 @_Z3red_zone(i1 zeroext %0, i32 %a, i32 %b) nounwind {
;; Check that cold blocks in functions with red zones aren't split.
; RED-ZONE-LABEL:        _Z3red_zone
; MFS-REDZONE-AARCH64-NOT:   _Z3red_zone.cold:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %x = alloca i32, align 4

  br i1 %0, label %2, label %3

2:                                                ; preds = %1
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  br label %4

3:                                                ; preds = %1
  store i32 %a, ptr %b.addr, align 4
  store i32 %b, ptr %a.addr, align 4
  br label %4

4:                                                ; preds = %3, %2
  %tmp = load i32, ptr %a.addr, align 4
  %tmp1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %tmp, %tmp1
  store i32 %add, ptr %x, align 4
  %tmp2 = load i32, ptr %x, align 4
  ret i32 %tmp2
}

declare i32 @bar()
declare i32 @baz()
declare i32 @bam()
declare i32 @qux()
