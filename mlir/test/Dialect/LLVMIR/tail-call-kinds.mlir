// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK: declare i32 @foo()
llvm.func @foo() -> i32

// CHECK-LABEL: @test_none
llvm.func @test_none() -> i32 {
  // CHECK-NEXT: call i32 @foo()
  %0 = llvm.call none @foo() : () -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @test_default
llvm.func @test_default() -> i32 {
  // CHECK-NEXT: call i32 @foo()
  %0 = llvm.call @foo() : () -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @test_musttail
llvm.func @test_musttail() -> i32 {
  // CHECK-NEXT: musttail call i32 @foo()
  %0 = llvm.call musttail @foo() : () -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @test_tail
llvm.func @test_tail() -> i32 {
  // CHECK-NEXT: tail call i32 @foo()
  %0 = llvm.call tail @foo() : () -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @test_notail
llvm.func @test_notail() -> i32 {
  // CHECK-NEXT: notail call i32 @foo()
  %0 = llvm.call notail @foo() : () -> i32
  llvm.return %0 : i32
}
