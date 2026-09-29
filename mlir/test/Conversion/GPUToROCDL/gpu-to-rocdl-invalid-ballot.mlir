// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx950' -split-input-file -verify-diagnostics

// -----

gpu.module @test_module {
  func.func @ballot_i8(%pred: i1) -> i8 {
    // expected-error @+1 {{failed to legalize operation 'gpu.ballot' that was explicitly marked illegal}}
    %0 = gpu.ballot %pred : i8
    func.return %0 : i8
  }
}

// -----

gpu.module @test_module {
  func.func @ballot_i16(%pred: i1) -> i16 {
    // expected-error @+1 {{failed to legalize operation 'gpu.ballot' that was explicitly marked illegal}}
    %0 = gpu.ballot %pred : i16
    func.return %0 : i16
  }
}

// -----

gpu.module @test_module {
  func.func @ballot_i48(%pred: i1) -> i48 {
    // expected-error @+1 {{failed to legalize operation 'gpu.ballot' that was explicitly marked illegal}}
    %0 = gpu.ballot %pred : i48
    func.return %0 : i48
  }
}

// -----

gpu.module @test_module {
  func.func @ballot_i128(%pred: i1) -> i128 {
    // expected-error @+1 {{failed to legalize operation 'gpu.ballot' that was explicitly marked illegal}}
    %0 = gpu.ballot %pred : i128
    func.return %0 : i128
  }
}
