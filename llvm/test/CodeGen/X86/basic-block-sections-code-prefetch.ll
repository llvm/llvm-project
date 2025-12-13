;; Check prefetch directives in basic block section profiles.
;;
;; Specify the bb sections profile:
; RUN: echo 'v1' > %t
; RUN: echo 'f _Z3foob' >> %t
; RUN: echo 't 0,0' >> %t
; RUN: echo 't 1,0' >> %t
; RUN: echo 't 1,1' >> %t
; RUN: echo 't 2,1' >> %t
; RUN: echo 't 3,0' >> %t
; RUN: echo 'f _Z3barv' >> %t
; RUN: echo 't 0,0' >> %t
; RUN: echo 't 21,1' >> %t
; RUN: echo 'f _Z3quxv' >> %t
; RUN: echo 't 0,0' >> %t
; RUN: echo 't 0,1' >> %t
;;
; RUN: llc < %s -mtriple=x86_64-pc-linux -asm-verbose=false -function-sections -basic-block-sections=%t -O0 | FileCheck %s

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
; CHECK:      _Z3foob:
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3foob_0_0
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_0_0:

9:                                                ; preds = %1
  %10 = call i32 @_Z3barv()
  store i32 %10, ptr %2, align 4
  br label %13
; CHECK:        .globl __llvm_prefetch_target__Z3foob_1_0
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_1_0:
; CHECK-NEXT:   callq _Z3barv@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3foob_1_1
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_1_1:

11:                                               ; preds = %1
  %12 = call i32 @_Z3bazv()
  store i32 %12, ptr %2, align 4
  br label %13
; CHECK:        callq _Z3bazv@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3foob_2_1
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_2_1:

13:                                               ; preds = %11, %9
  %14 = load i32, ptr %2, align 4
  ret i32 %14
; CHECK:      .LBB0_3:
; CHECK-NEXT:   .globl	__llvm_prefetch_target__Z3foob_3_0
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_3_0:
}

define weak i32 @_Z3barv() nounwind {
  %1 = call i32 @_Z3bazv()
  ret i32 %1
; CHECK:      _Z3barv:
; CHECK-NEXT:   .weak __llvm_prefetch_target__Z3barv_0_0
; CHECK-NEXT: __llvm_prefetch_target__Z3barv_0_0:
; CHECK:        callq _Z3bazv@PLT
}

define internal i32 @_Z3quxv() nounwind {
  %1 = call i32 @_Z3bazv()
  ret i32 %1
; CHECK:      _Z3quxv:
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3quxv_0_0
; CHECK-NEXT: __llvm_prefetch_target__Z3quxv_0_0:
; CHECK:        callq _Z3bazv@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3quxv_0_1
; CHECK-NEXT: __llvm_prefetch_target__Z3quxv_0_1:
}

declare i32 @_Z3bazv() #1
