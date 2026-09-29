// RUN: mlir-opt %s -inline -verify-diagnostics -split-input-file

module {
  llvm.func @gen_broadcast_scalar1d(%arg0: i32) -> i32 attributes {llvm.emit_c_interface} {
    %0 = "test.broadcast_bounds_mismatch1"(%arg0) : (i32) -> i32
    // Missing terminator here
  }

  llvm.func @_mlir_ciface_gen_broadcast_scalar1d(%arg0: i32) -> i32 attributes {llvm.emit_c_interface} {
    // expected-error @+1 {{not all uses of call were replaced}}
    %0 = llvm.call @gen_broadcast_scalar1d(%arg0) : (i32) -> i32
    llvm.return %0 : i32
  }
}