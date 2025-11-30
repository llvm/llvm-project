// RUN: mlir-opt -gpu-kernel-outlining -verify-diagnostics -split-input-file %s

module attributes {gpu.container_module} {
  func.func @kernel_crash() {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
               threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
      // expected-error@+1 {{failed to outline gpu kernel: symbol 'unknown_func' not found}}
      "test.op"() {symbol = @unknown_func} : () -> ()
      gpu.terminator
    }
    return
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @kernel_invalid_ref() {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
               threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
      // expected-error@+1 {{failed to outline gpu kernel: found invalid symbol reference: @nested::@ref}}
      "test.op"() {symbol = @nested::@ref} : () -> ()
      gpu.terminator
    }
    return
  }
}
