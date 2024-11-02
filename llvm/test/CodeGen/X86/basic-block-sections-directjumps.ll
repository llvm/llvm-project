; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=all -unique-basic-block-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu  -function-sections -basic-block-sections=all -unique-basic-block-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS

define void @_Z3bazb(i1 zeroext) {
  %2 = alloca i8, align 1
  %3 = zext i1 %0 to i8
  store i8 %3, ptr %2, align 1
  %4 = load i8, ptr %2, align 1
  %5 = trunc i8 %4 to i1
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = call i32 @_Z3barv()
  %8 = trunc i32 %7 to i1
  br i1 %8, label %11, label %9

9:                                                ; preds = %1
  %10 = call i32 @_Z3foov()
  br label %11

11:                                               ; preds = %9, %6
  ret void
}

declare i32 @_Z3barv() #1

declare i32 @_Z3foov() #1


; LINUX-SECTIONS: .section        .text._Z3bazb,"ax",@progbits
; LINUX-SECTIONS: _Z3bazb:
; LINUX-SECTIONS: jmp _Z3bazb.__part.1
; LINUX-SECTIONS: .section        .text._Z3bazb._Z3bazb.__part.1,"ax",@progbits
; LINUX-SECTIONS: _Z3bazb.__part.1:
; LINUX-SECTIONS: jmp _Z3bazb.__part.2
; LINUX-SECTIONS: .section        .text._Z3bazb._Z3bazb.__part.2,"ax",@progbits
; LINUX-SECTIONS: _Z3bazb.__part.2:
; LINUX-SECTIONS: jmp _Z3bazb.__part.3
