// RUN: mlir-opt %s -convert-gpu-to-nvvm='allowed-dialects=test' -verify-diagnostics

// expected-error @+1 {{dialect does not implement ConvertToLLVMPatternInterface: test}}
gpu.module @test_module_1 {
  func.func @test(%0 : index) -> index {
    %1 = test.increment %0 : index
    func.return %1 : index
  }
}

