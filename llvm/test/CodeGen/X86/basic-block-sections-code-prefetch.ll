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

define void @_Z3foob(i1 %arg) nounwind {
  br i1 %arg, label %cond.true, label %cond.false
; CHECK:      _Z3foob:
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3foob_0_0
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_0_0:

cond.true:                                           ; preds = %1
  call i32 @_Z3barv()
  br label %end
; CHECK:        .globl __llvm_prefetch_target__Z3foob_1_0
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_1_0:
; CHECK-NEXT:   callq _Z3barv@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3foob_1_1
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_1_1:

cond.false:                                          ; preds = %1
  call i32 @_Z3bazv()
  br label %end
; CHECK:        callq _Z3bazv@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3foob_2_1
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_2_1:

end:                                             ; preds = %11, %9
  ret void
; CHECK:      .LBB0_3:
; CHECK-NEXT:   .globl	__llvm_prefetch_target__Z3foob_3_0
; CHECK-NEXT: __llvm_prefetch_target__Z3foob_3_0:
}

define weak i32 @_Z3barv() nounwind {
  %call = call i32 @_Z3bazv()
  ret i32 %call
; CHECK:      _Z3barv:
; CHECK-NEXT:   .weak __llvm_prefetch_target__Z3barv_0_0
; CHECK-NEXT: __llvm_prefetch_target__Z3barv_0_0:
; CHECK:        callq _Z3bazv@PLT
}

define internal i32 @_Z3quxv() nounwind {
  %call = call i32 @_Z3bazv()
  ret i32 %call
; CHECK:      _Z3quxv:
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3quxv_0_0
; CHECK-NEXT: __llvm_prefetch_target__Z3quxv_0_0:
; CHECK:        callq _Z3bazv@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target__Z3quxv_0_1
; CHECK-NEXT: __llvm_prefetch_target__Z3quxv_0_1:
}

declare i32 @_Z3bazv() #1
declare i32 @_Z3dummy() #2
