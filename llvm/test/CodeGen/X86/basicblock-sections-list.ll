; Check the basic block sections list option.
; RUN: echo '!_Z3foob' > %t
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basicblock-sections=%t -unique-bb-section-names | FileCheck %s -check-prefix=LINUX-SECTIONS

define i32 @_Z3foob(i1 zeroext %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i8, align 1
  %4 = zext i1 %0 to i8
  store i8 %4, i8* %3, align 1
  %5 = load i8, i8* %3, align 1
  %6 = trunc i8 %5 to i1
  %7 = zext i1 %6 to i32
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %11

9:                                                ; preds = %1
  %10 = call i32 @_Z3barv()
  store i32 %10, i32* %2, align 4
  br label %13

11:                                               ; preds = %1
  %12 = call i32 @_Z3bazv()
  store i32 %12, i32* %2, align 4
  br label %13

13:                                               ; preds = %11, %9
  %14 = load i32, i32* %2, align 4
  ret i32 %14
}

declare i32 @_Z3barv() #1
declare i32 @_Z3bazv() #1

define i32 @_Z3zipb(i1 zeroext %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i8, align 1
  %4 = zext i1 %0 to i8
  store i8 %4, i8* %3, align 1
  %5 = load i8, i8* %3, align 1
  %6 = trunc i8 %5 to i1
  %7 = zext i1 %6 to i32
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %11

9:                                                ; preds = %1
  %10 = call i32 @_Z3barv()
  store i32 %10, i32* %2, align 4
  br label %13

11:                                               ; preds = %1
  %12 = call i32 @_Z3bazv()
  store i32 %12, i32* %2, align 4
  br label %13

13:                                               ; preds = %11, %9
  %14 = load i32, i32* %2, align 4
  ret i32 %14
}

; LINUX-SECTIONS: .section        .text._Z3foob,"ax",@progbits
; LINUX-SECTIONS: _Z3foob:
; LINUX-SECTIONS: .section        .text._Z3foob.a.BB._Z3foob,"ax",@progbits,unique,1
; LINUX-SECTIONS: a.BB._Z3foob:
; LINUX-SECTIONS: .section        .text._Z3foob.aa.BB._Z3foob,"ax",@progbits,unique,2
; LINUX-SECTIONS: aa.BB._Z3foob:
; LINUX-SECTIONS: .section        .text._Z3foob.raa.BB._Z3foob,"ax",@progbits,unique,3
; LINUX-SECTIONS: raa.BB._Z3foob:

; LINUX-SECTIONS: .section        .text._Z3zipb,"ax",@progbits
; LINUX-SECTIONS: _Z3zipb:
; LINUX-SECTIONS-NOT: .section        .text._Z3zipb.a.BB._Z3zipb,"ax",@progbits,unique,1
; LINUX-SECTIONS-NOT: a.BB._Z3zipb:
; LINUX-SECTIONS-NOT: .section        .text._Z3zipb.aa.BB._Z3zipb,"ax",@progbits,unique,2
; LINUX-SECTIONS-NOT: aa.BB._Z3zipb:
; LINUX-SECTIONS-NOT: .section        .text._Z3zipb.raa.BB._Z3zipb,"ax",@progbits,unique,3
; LINUX-SECTIONS-NOT: raa.BB._Z3zipb:
