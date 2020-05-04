; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basicblock-sections=all -unique-bb-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu  -function-sections -basicblock-sections=all -unique-bb-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS

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
; LINUX-SECTIONS: .section        .text._Z3bazb._Z3bazb.1,"ax",@progbits
; LINUX-SECTIONS: _Z3bazb.1:
; LINUX-SECTIONS: .Ltmp0:
; LINUX-SECTIONS-NEXT: .size   _Z3bazb.1, .Ltmp0-_Z3bazb.1
; LINUX-SECTIONS: .section        .text._Z3bazb._Z3bazb.2,"ax",@progbits
; LINUX-SECTIONS: _Z3bazb.2:
; LINUX-SECTIONS: .Ltmp1:
; LINUX-SECTIONS-NEXT: .size   _Z3bazb.2, .Ltmp1-_Z3bazb.2
