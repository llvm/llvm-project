; REQUIRES: asserts
; RUN: opt -passes="ipsccp<func-spec>,inline,instcombine,simplifycfg" -S \
; RUN:     -funcspec-min-function-size=23 -funcspec-max-iters=100 \
; RUN:     -debug-only=function-specialization < %s 2>&1 | FileCheck %s

; Make sure the number of specializations created are not
; linear to the number of iterations (funcspec-max-iters).

; CHECK: FnSpecialization: Created 4 specializations in module

@Global = internal constant i32 1, align 4

define internal void @recursiveFunc(ptr readonly %arg) {
  %temp = alloca i32, align 4
  %arg.load = load i32, ptr %arg, align 4
  %arg.cmp = icmp slt i32 %arg.load, 10000
  br i1 %arg.cmp, label %loop1, label %ret.block

loop1:
  br label %loop2

loop2:
  br label %loop3

loop3:
  br label %loop4

loop4:
  br label %block6

block6:
  call void @print_val(i32 %arg.load)
  %arg.add = add nsw i32 %arg.load, 1
  store i32 %arg.add, ptr %temp, align 4
  call void @recursiveFunc(ptr %temp)
  br label %loop4.end

loop4.end:
  %exit_cond1 = call i1 @exit_cond()
  br i1 %exit_cond1, label %loop4, label %loop3.end

loop3.end:
  %exit_cond2 = call i1 @exit_cond()
  br i1 %exit_cond2, label %loop3, label %loop2.end

loop2.end:
  %exit_cond3 = call i1 @exit_cond()
  br i1 %exit_cond3, label %loop2, label %loop1.end

loop1.end:
  %exit_cond4 = call i1 @exit_cond()
  br i1 %exit_cond4, label %loop1, label %ret.block

ret.block:
  ret void
}

define i32 @main() {
  call void @recursiveFunc(ptr @Global)
  ret i32 0
}

declare dso_local void @print_val(i32)
declare dso_local i1 @exit_cond()
