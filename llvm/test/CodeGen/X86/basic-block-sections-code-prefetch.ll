;; Check prefetch directives in basic block section profiles.
;;
;; Specify the bb sections profile:
; RUN: echo 'v1' > %t
; RUN: echo 'f foo' >> %t
; RUN: echo 't 0,0' >> %t
; RUN: echo 't 1,0' >> %t
; RUN: echo 't 1,1' >> %t
; RUN: echo 't 2,1' >> %t
; RUN: echo 't 3,0' >> %t
; RUN: echo 'f bar' >> %t
; RUN: echo 't 0,0' >> %t
; RUN: echo 't 21,1' >> %t
; RUN: echo 'f qux' >> %t
; RUN: echo 't 0,0' >> %t
; RUN: echo 't 0,1' >> %t
;;
; RUN: llc < %s -mtriple=x86_64-pc-linux -asm-verbose=false -function-sections -basic-block-sections=%t -O0 | FileCheck %s

define void @foo(i1 %arg) nounwind {
  br i1 %arg, label %cond.true, label %cond.false
; CHECK:      foo:
; CHECK-NEXT:   .globl __llvm_prefetch_target_foo_0_0
; CHECK-NEXT: __llvm_prefetch_target_foo_0_0:

cond.true:                                           ; preds = %1
  call i32 @bar()
  br label %end
; CHECK:        .globl __llvm_prefetch_target_foo_1_0
; CHECK-NEXT: __llvm_prefetch_target_foo_1_0:
; CHECK-NEXT:   callq bar@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target_foo_1_1
; CHECK-NEXT: __llvm_prefetch_target_foo_1_1:

cond.false:                                          ; preds = %1
  call i32 @baz()
  br label %end
; CHECK:        callq baz@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target_foo_2_1
; CHECK-NEXT: __llvm_prefetch_target_foo_2_1:

end:                                             ; preds = %11, %9
  ret void
; CHECK:      .LBB0_3:
; CHECK-NEXT:   .globl	__llvm_prefetch_target_foo_3_0
; CHECK-NEXT: __llvm_prefetch_target_foo_3_0:
}

define weak i32 @bar() nounwind {
  %call = call i32 @baz()
  ret i32 %call
; CHECK:      bar:
; CHECK-NEXT:   .weak __llvm_prefetch_target_bar_0_0
; CHECK-NEXT: __llvm_prefetch_target_bar_0_0:
; CHECK:        callq baz@PLT
}

define internal i32 @qux() nounwind {
  %call = call i32 @baz()
  ret i32 %call
; CHECK:      qux:
; CHECK-NEXT:   .globl __llvm_prefetch_target_qux_0_0
; CHECK-NEXT: __llvm_prefetch_target_qux_0_0:
; CHECK:        callq baz@PLT
; CHECK-NEXT:   .globl __llvm_prefetch_target_qux_0_1
; CHECK-NEXT: __llvm_prefetch_target_qux_0_1:
}

declare i32 @baz()
declare i32 @dummy()
