; RUN: opt -S -passes=spirv-merge-region-exits -mtriple=spirv-unknown-vulkan-compute < %s | FileCheck %s

; A loop with two distinct exit targets (exit_a, exit_b). The pass
; should merge them into a single new.exit block dispatched via a
; switch on a stored discriminator value.

define spir_func i32 @two_exits(i1 %ca, i1 %cb) #0 {
; CHECK-LABEL: define spir_func i32 @two_exits(
; CHECK:       entry:
; CHECK:         [[REG:%.*]] = alloca i32
; CHECK:       loop:
; CHECK:         store i32 {{.*}}, ptr [[REG]]
; CHECK:         br i1 %ca, label %new.exit, label %body
; CHECK:       body:
; CHECK:         store i32 {{.*}}, ptr [[REG]]
; CHECK:         br i1 %cb, label %new.exit, label %loop
; CHECK:       new.exit:
; CHECK:         [[V:%.*]] = load i32, ptr [[REG]]
; CHECK:         switch i32 [[V]], label %exit_a [
; CHECK:           i32 {{.*}}, label %exit_b
; CHECK:         ]
entry:
  %t0 = call token @llvm.experimental.convergence.entry()
  br label %loop

loop:
  %t1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t0) ]
  br i1 %ca, label %exit_a, label %body

body:
  br i1 %cb, label %exit_b, label %loop

exit_a:
  ret i32 1

exit_b:
  ret i32 2
}

declare token @llvm.experimental.convergence.entry() #1
declare token @llvm.experimental.convergence.loop() #1

attributes #0 = { convergent noinline nounwind }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
