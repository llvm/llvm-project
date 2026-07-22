// RUN: mlir-opt %s -test-one-shot-module-bufferize -verify-diagnostics -split-input-file

#custom_buffer = #test.tensor_encoding<"custom_buffer">

module {
  // expected-error @below {{cannot apply buffer layout to buffer type '!test.test_memref<[1], !llvm.array<1 x i32>>'}}
  // expected-error @below {{failed to bufferize op}}
  func.func private @custom_buffer_with_layout(
      tensor<1x!llvm.array<1 x i32>, #custom_buffer>
        {bufferization.buffer_layout = affine_map<(d0) -> (d0)>})
}

// -----

#custom_buffer = #test.tensor_encoding<"custom_buffer">

module {
  func.func private @custom_buffer_callee(
      tensor<1xf32, #custom_buffer>)

  func.func @custom_buffer_caller() {
    %tensor = "test.create_tensor_op"() : () -> tensor<1xf32, #custom_buffer>
    // expected-error @below {{cannot reconcile buffer types 'memref<1xf32, #test.memref_layout<"custom_buffer">>' and '!test.test_memref<[1], f32>'}}
    // expected-error @below {{failed to bufferize op}}
    func.call @custom_buffer_callee(%tensor)
        : (tensor<1xf32, #custom_buffer>) -> ()
    return
  }
}
