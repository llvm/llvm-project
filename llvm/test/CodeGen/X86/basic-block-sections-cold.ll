;; Check if basic blocks that don't get unique sections are placed in cold sections.
;; Basic block with id 1 and 2 must be in the cold section.
;;
;; Profile for version 0
; RUN: echo '!_Z3bazb' > %t1
; RUN: echo '!!0' >> %t1
;;
;; Profile for version 1
; RUN: echo 'v1' > %t2
; RUN: echo 'f _Z3bazb' >> %t2
; RUN: echo 'c 0' >> %t2
;;
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t1 -unique-basic-block-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t2 -unique-basic-block-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS
; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=%t1 -unique-basic-block-section-names -bbsections-cold-text-prefix=".text.unlikely." | FileCheck %s -check-prefix=LINUX-SPLIT

define void @_Z3bazb(i1 zeroext %0) nounwind {
  br i1 %0, label %2, label %4

2:                                                ; preds = %1
  %3 = call i32 @_Z3barv()
  br label %6

4:                                                ; preds = %1
  %5 = call i32 @_Z3foov()
  br label %6

6:                                                ; preds = %2, %4
  ret void
}

declare i32 @_Z3barv() #1

declare i32 @_Z3foov() #1

; LINUX-SECTIONS: .section        .text.hot._Z3bazb,"ax",@progbits
; LINUX-SECTIONS: _Z3bazb:
; Check that the basic block with id 1 doesn't get a section.
; LINUX-SECTIONS-NOT: .section        .text{{.*}}._Z3bazb.1,"ax",@progbits,unique
; Check that a single cold section is started here and id 1 and 2 blocks are placed here.
; LINUX-SECTIONS: .section	.text.split._Z3bazb,"ax",@progbits
; LINUX-SECTIONS: _Z3bazb.cold:
; LINUX-SECTIONS-NOT: .section        .text.hot._Z3bazb._Z3bazb.2,"ax",@progbits,unique
; LINUX-SECTIONS: .LBB0_2:
; LINUX-SECTIONS: .size   _Z3bazb, .Lfunc_end{{[0-9]}}-_Z3bazb

; LINUX-SPLIT:      .section	.text.unlikely._Z3bazb,"ax",@progbits
; LINUX-SPLIT-NEXT: _Z3bazb.cold:
; LINUX-SPLIT-NEXT:   callq _Z3barv
; LINUX-SPLIT:      .LBB0_2:
; LINUX-SPLIT:      .LBB_END0_2:
