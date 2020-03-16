; Fine-grained basic block sections, subset of basic blocks in a function.
; Only basic block with id 2 must get a section.
; RUN: echo '!_Z3bazb' > %t
; RUN: echo '!!2' >> %t
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basicblock-sections=%t -unique-bb-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS

define void @_Z3bazb(i1 zeroext) {
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
; LINUX-SECTIONS-NOT: .section        .text._Z3bazb.r.BB._Z3bazb,"ax",@progbits,unique
; LINUX-SECTIONS: r.BB._Z3bazb:
; LINUX-SECTIONS: .section        .text._Z3bazb.rr.BB._Z3bazb,"ax",@progbits,unique
; LINUX-SECTIONS: rr.BB._Z3bazb:
; LINUX-SECTIONS: .size   rr.BB._Z3bazb, .Ltmp1-rr.BB._Z3bazb
