; Check if basic blocks that don't get unique sections are placed in cold sections.
; Basic block with id 1 and 2 must be in the cold section.
; RUN: echo '!_Z3bazb' > %t
; RUN: echo '!!0' >> %t
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t -unique-basic-block-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS

define void @_Z3bazb(i1 zeroext) nounwind {
  %2 = alloca i8, align 1
  %3 = zext i1 %0 to i8
  store i8 %3, i8* %2, align 1
  %4 = load i8, i8* %2, align 1
  %5 = trunc i8 %4 to i1
  br i1 %5, label %6, label %8

6:                                                ; preds = %1
  %7 = call i32 @_Z3barv()
  br label %10

8:                                                ; preds = %1
  %9 = call i32 @_Z3foov()
  br label %10

10:                                               ; preds = %8, %6
  ret void
}

declare i32 @_Z3barv() #1

declare i32 @_Z3foov() #1

; LINUX-SECTIONS: .section        .text._Z3bazb,"ax",@progbits
; LINUX-SECTIONS: _Z3bazb:
; Check that the basic block with id 1 doesn't get a section.
; LINUX-SECTIONS-NOT: .section        .text._Z3bazb._Z3bazb.1,"ax",@progbits,unique
; Check that a single cold section is started here and id 1 and 2 blocks are placed here.
; LINUX-SECTIONS: .section	.text.unlikely._Z3bazb,"ax",@progbits
; LINUX-SECTIONS: _Z3bazb.cold:
; LINUX-SECTIONS-NOT: .section        .text._Z3bazb._Z3bazb.2,"ax",@progbits,unique
; LINUX-SECTIONS: .LBB0_2:
; LINUX-SECTIONS: .size   _Z3bazb, .Lfunc_end{{[0-9]}}-_Z3bazb
