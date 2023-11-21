; RUN: opt -passes="ipsccp<func-spec>" -force-specialization -funcspec-max-iters=2 -S < %s | FileCheck %s

; Volatile store preventing recursive specialisation:
;
; CHECK:     @recursiveFunc.specialized.1
; CHECK-NOT: @recursiveFunc.specialized.2

@Global = internal constant i32 1, align 4

define internal void @recursiveFunc(ptr nocapture readonly %arg) {
  %temp = alloca i32, align 4
  %arg.load = load i32, ptr %arg, align 4
  %arg.cmp = icmp slt i32 %arg.load, 4
  br i1 %arg.cmp, label %block6, label %ret.block

block6:
  call void @print_val(i32 %arg.load)
  %arg.add = add nsw i32 %arg.load, 1
  store volatile i32 %arg.add, ptr %temp, align 4
  call void @recursiveFunc(ptr nonnull %temp)
  br label %ret.block

ret.block:
  ret void
}

define i32 @main() {
  call void @recursiveFunc(ptr nonnull @Global)
  ret i32 0
}

declare dso_local void @print_val(i32)
