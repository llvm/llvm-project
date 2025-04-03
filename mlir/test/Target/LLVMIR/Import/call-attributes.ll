; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

declare void @f()

; CHECK-LABEL: @test_call_noinline
; CHECK: llvm.call @f() {no_inline} : () -> ()
define void @test_call_noinline() {
  call void @f() #0
  ret void
}

attributes #0 = { noinline }

// -----

declare void @f()

; CHECK-LABEL: @test_call_alwaysinline
; CHECK: llvm.call @f() {always_inline} : () -> ()
define void @test_call_alwaysinline() {
  call void @f() #0 
  ret void
}

attributes #0 = { alwaysinline }
