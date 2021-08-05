; RUN: opt -function-specialization -force-function-specialization -func-specialization-max-iters=2 -S < %s | FileCheck %s

; Alloca is not an integer type:
;
; CHECK:     @recursiveFunc.1
; CHECK-NOT: @recursiveFunc.2

@Global = internal constant i32 1, align 4

define internal void @recursiveFunc(i32* nocapture readonly %arg) {
  %temp = alloca float, align 4
  %arg.load = load i32, i32* %arg, align 4
  %arg.cmp = icmp slt i32 %arg.load, 4
  br i1 %arg.cmp, label %block6, label %ret.block

block6:
  call void @print_val(i32 %arg.load)
  %arg.add = add nsw i32 %arg.load, 1
  %bc = bitcast float* %temp to i32*
  call void @recursiveFunc(i32* nonnull %bc)
  br label %ret.block

ret.block:
  ret void
}

define i32 @main() {
  call void @recursiveFunc(i32* nonnull @Global)
  ret i32 0
}

declare dso_local void @print_val(i32)
