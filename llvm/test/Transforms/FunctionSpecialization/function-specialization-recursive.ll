; RUN: opt -function-specialization -force-function-specialization -func-specialization-max-iters=2 -inline -instcombine -S < %s | FileCheck %s --check-prefix=ITERS2
; RUN: opt -function-specialization -force-function-specialization -func-specialization-max-iters=3 -inline -instcombine -S < %s | FileCheck %s --check-prefix=ITERS3
; RUN: opt -function-specialization -force-function-specialization -func-specialization-max-iters=4 -inline -instcombine -S < %s | FileCheck %s --check-prefix=ITERS4

@Global = internal constant i32 1, align 4

define internal void @recursiveFunc(i32* nocapture readonly %arg) {
  %temp = alloca i32, align 4
  %arg.load = load i32, i32* %arg, align 4
  %arg.cmp = icmp slt i32 %arg.load, 4
  br i1 %arg.cmp, label %block6, label %ret.block

block6:
  call void @print_val(i32 %arg.load)
  %arg.add = add nsw i32 %arg.load, 1
  store i32 %arg.add, i32* %temp, align 4
  call void @recursiveFunc(i32* nonnull %temp)
  br label %ret.block

ret.block:
  ret void
}

; ITERS2:  @funcspec.arg.3 = internal constant i32 3
; ITERS3:  @funcspec.arg.5 = internal constant i32 4

define i32 @main() {
; ITERS2-LABEL: @main(
; ITERS2-NEXT:    call void @print_val(i32 1)
; ITERS2-NEXT:    call void @print_val(i32 2)
; ITERS2-NEXT:    call void @recursiveFunc(i32* nonnull @funcspec.arg.3)
; ITERS2-NEXT:    ret i32 0
;
; ITERS3-LABEL: @main(
; ITERS3-NEXT:    call void @print_val(i32 1)
; ITERS3-NEXT:    call void @print_val(i32 2)
; ITERS3-NEXT:    call void @print_val(i32 3)
; ITERS3-NEXT:    call void @recursiveFunc(i32* nonnull @funcspec.arg.5)
; ITERS3-NEXT:    ret i32 0
;
; ITERS4-LABEL: @main(
; ITERS4-NEXT:    call void @print_val(i32 1)
; ITERS4-NEXT:    call void @print_val(i32 2)
; ITERS4-NEXT:    call void @print_val(i32 3)
; ITERS4-NEXT:    ret i32 0
;
  call void @recursiveFunc(i32* nonnull @Global)
  ret i32 0
}

declare dso_local void @print_val(i32)
