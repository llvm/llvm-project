;; Check that specifying the function in the basic block sections profile
;; without any other directives is a noop.
;;
;; Specify the bb sections profile:
; RUN: echo 'v1' > %t
; RUN: echo 'f _Z3foob' >> %t
;;
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t  > %t.bbsections
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections > %t.orig
; RUN: diff -u %t.orig %t.bbsections

define i32 @_Z3foob(i1 zeroext %0) nounwind {
  %2 = alloca i32, align 4
  %3 = alloca i8, align 1
  %4 = zext i1 %0 to i8
  store i8 %4, ptr %3, align 1
  %5 = load i8, ptr %3, align 1
  %6 = trunc i8 %5 to i1
  %7 = zext i1 %6 to i32
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %11

9:                                                ; preds = %1
  %10 = call i32 @_Z3barv()
  store i32 %10, ptr %2, align 4
  br label %13

11:                                               ; preds = %1
  %12 = call i32 @_Z3bazv()
  store i32 %12, ptr %2, align 4
  br label %13

13:                                               ; preds = %11, %9
  %14 = load i32, ptr %2, align 4
  ret i32 %14
}

declare i32 @_Z3barv() #1
declare i32 @_Z3bazv() #1
