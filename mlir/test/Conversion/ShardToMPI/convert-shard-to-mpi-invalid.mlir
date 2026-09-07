// RUN: mlir-opt %s -convert-shard-to-mpi -verify-diagnostics

module {
  func.func @vector_ops(%arg0: memref<4xf32>) {
    %mask = vector.constant_mask [8] : vector<8xi1>
    %zero = arith.constant 0.000000e+00 : f32
    %broadcast = vector.broadcast %zero : f32 to vector<8xf32>
    %reduction = vector.mask %mask {
      vector.multi_reduction <add>, %broadcast, %zero [0]
        : vector<8xf32> to f32
    } : vector<8xi1> -> f32
    %result = arith.addf %reduction, %zero : f32
    return
  }

  func.func @unsupported_send(%arg0: tensor<4xf32>) {
    // expected-error@+1 {{failed to legalize operation 'shard.send' that was explicitly marked illegal}}
    %send = shard.send %arg0 on @grid destination = []
      : (tensor<4xf32>) -> tensor<4xf32>
    return
  }

  shard.grid @grid(shape = 2)
}
