// RUN: mlir-opt --test-emulate-narrow-int --verify-diagnostics %s

module {
  func.func @test() {
    %alloca = memref.alloca() : memref<vector<[2]x2xi1>>
    // expected-error @+1 {{failed to legalize operation 'memref.load'}}
    %val = memref.load %alloca[] : memref<vector<[2]x2xi1>>
    return
  }
}