; Check the basic block sections labels option
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=labels | FileCheck %s -check-prefix=LINUX-LABELS

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

; LINUX-LABELS: .section
; LINUX-LABELS: _Z3bazb:
; LINUX-LABELS-NOT: .section
; LINUX-LABELS: r.BB._Z3bazb:
; LINUX-LABELS-NOT: .section
; LINUX-LABELS: rr.BB._Z3bazb:
