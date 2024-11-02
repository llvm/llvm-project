// RUN: mlir-opt %s \
// RUN:     -transform-interpreter \
// RUN:     -test-transform-dialect-erase-schedule \
// RUN: | FileCheck %s

memref.global "private" @bufferLhsGlobal : memref<64x32xf32, #gpu.address_space<workgroup>>
memref.global "private" @bufferRhsGlobal : memref<8x32xf32, #gpu.address_space<workgroup>>

// CHECK-LABEL: func.func @main()
func.func @main() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index

  %0 = gpu.wait async
  %memref, %asyncToken = gpu.alloc async [%0] () : memref<64x32xf32>
  %memref_1, %asyncToken_2 = gpu.alloc async [%0] () : memref<8x32xf32>

  //      CHECK: %[[M1:.*]] = memref.cast %{{.*}} : memref<64x32xf32> to memref<*xf32>
  //      CHECK: %[[c64:.*]] = arith.constant 64 : index
  //      CHECK: %[[c32:.*]] = arith.constant 32 : index
  //      CHECK: %[[D1:.*]] = nvgpu.tma.create.descriptor %[[M1]] box[%[[c64]], %[[c32]]]
  // CHECK-SAME:   : memref<*xf32> -> <tensor = memref<64x32xf32, #gpu.address_space<workgroup>>, swizzle = none, l2promo = none, oob = zero, interleave = none>
  //      CHECK: %[[cast_2:.*]] = memref.cast %memref_0 : memref<8x32xf32> to memref<*xf32>
  //      CHECK: %[[c8_2:.*]] = arith.constant 8 : index
  //      CHECK: %[[c32_2:.*]] = arith.constant 32 : index
  //      CHECK: %[[D2:.*]] = nvgpu.tma.create.descriptor %cast_2 box[%[[c8_2]], %[[c32_2]]]
  // CHECK-SAME:   : memref<*xf32> -> <tensor = memref<8x32xf32, #gpu.address_space<workgroup>>, swizzle = none, l2promo = none, oob = zero, interleave = none>
  // CHECK: gpu.launch
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
            threads(%tx, %ty, %tz) in (%block_x = %c128, %block_y = %c1, %block_z = %c1) {
    //      CHECK: %[[G1:.*]] = memref.get_global @bufferLhsGlobal : memref<64x32xf32, #gpu.address_space<workgroup>>
    //      CHECK: %[[G2:.*]] = memref.get_global @bufferRhsGlobal : memref<8x32xf32, #gpu.address_space<workgroup>>
    %out = memref.get_global @bufferLhsGlobal : memref<64x32xf32, #gpu.address_space<workgroup>>
    %out_1 = memref.get_global @bufferRhsGlobal : memref<8x32xf32, #gpu.address_space<workgroup>>

    //      CHECK: %[[B:.*]] = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>
    //      CHECK: nvgpu.mbarrier.init %[[B]][%{{.*}}], %{{.*}} : <memorySpace = #gpu.address_space<workgroup>
    //      CHECK: gpu.barrier
    //
    //      CHECK: %[[c0:.*]] = arith.constant 0 : index
    //      CHECK: %[[TIDX:.*]] = gpu.thread_id  x
    //      CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[TIDX]], %[[c0]] : index
    //
    //      CHECK: scf.if %[[CMP]] {
    //
    //      CHECK:   %[[c0_7:.*]] = arith.constant 0 : index
    //      CHECK:   nvgpu.tma.async.load %[[D1]][%[[c0_7]], %[[c0_7]]], %[[B]][%{{.*}}] to %[[G1]]
    // CHECK-SAME:     : <tensor = memref<64x32xf32, #gpu.address_space<workgroup>>,
    // CHECK-SAME:        swizzle = none, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>
    // CHECK-SAME:     -> memref<64x32xf32, #gpu.address_space<workgroup>>
    //
    //      CHECK:   %[[c0_8:.*]] = arith.constant 0 : index
    //      CHECK:   nvgpu.tma.async.load %[[D2]][%[[c0_8]], %[[c0_8]]], %[[B]][%{{.*}}] to %[[G2]]
    // CHECK-SAME:     : <tensor = memref<8x32xf32, #gpu.address_space<workgroup>>,
    // CHECK-SAME:         swizzle = none, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>
    // CHECK-SAME:    -> memref<8x32xf32, #gpu.address_space<workgroup>>
    //
    //      CHECK:   %[[c9216:.*]] = arith.constant 9216 : index
    //      CHECK:   nvgpu.mbarrier.arrive.expect_tx %[[B]][%{{.*}}], %[[c9216]] : <memorySpace = #gpu.address_space<workgroup>
    //      CHECK: } else {
    //      CHECK:   %[[c0_7:.*]] = arith.constant 0 : index
    //      CHECK:   nvgpu.mbarrier.arrive.expect_tx %[[B]][%{{.*}}], %[[c0_7]] : <memorySpace = #gpu.address_space<workgroup>
    //      CHECK: }
    //
    //      CHECK: %[[c0_6:.*]] = arith.constant 0 : index
    //      CHECK: %[[c10000000:.*]] = arith.constant 10000000 : index
    //      CHECK: nvgpu.mbarrier.try_wait.parity %[[B]][%{{.*}}], %[[c0_6]], %[[c10000000]] : <memorySpace = #gpu.address_space<workgroup>

    /// Both copies are matched and end up in the same async group.
    linalg.copy ins(%memref: memref<64x32xf32>) outs(%out: memref<64x32xf32, #gpu.address_space<workgroup>>)
    linalg.copy ins(%memref_1: memref<8x32xf32>) outs(%out_1: memref<8x32xf32, #gpu.address_space<workgroup>>)

    gpu.terminator
  }

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %copy = transform.structured.match ops{["linalg.copy"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.rewrite_copy_as_tma %copy  : (!transform.any_op) -> ()
    transform.yield
  }
}
