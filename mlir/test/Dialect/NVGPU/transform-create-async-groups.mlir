// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file --verify-diagnostics | FileCheck %s

// Check that we produce async copies from the vector.transfer_xxx operations.
builtin.module {
  // CHECK-LABEL: @copies_to_asyncs
  func.func @copies_to_asyncs(%a: memref<1024x1024xf32>) {
    %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    // Make sure we emit the bypassL1.
    // CHECK: %[[CP0:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 4  {bypassL1} :
    %1 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
    vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
    // CHECK-NOT: nvgpu.device_async_create_group

    // CHECK: %[[CP1:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 1
    %2 = vector.transfer_read %a[%c0, %c4], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
    vector.transfer_write %2, %0[%c0, %c4, %c0] {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[G:.*]] = nvgpu.device_async_create_group %[[CP0]], %[[CP1]]
    // CHECK: nvgpu.device_async_wait %[[G]]
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %top_level_func {bypass_l1} : (!transform.any_op) -> (!transform.any_op)
  }
}

// -----

// Check that we properly take `bypass_l1 = false` into account.
// I.e., we shouldn't be generating bypassL1 attributes.
builtin.module {
  // CHECK-LABEL: @copies_to_asyncs_no_mma
  func.func @copies_to_asyncs_no_mma(%a: memref<1024x1024xf32>) {
    %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    // Make sure we don't emit the bypassL1.
    // CHECK: %[[CP0:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 4 :
    %1 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
    vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
    // CHECK-NOT: nvgpu.device_async_create_group

    // CHECK: %[[CP1:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 1 :
    %2 = vector.transfer_read %a[%c0, %c4], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
    vector.transfer_write %2, %0[%c0, %c4, %c0] {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[G:.*]] = nvgpu.device_async_create_group %[[CP0]], %[[CP1]]
    // CHECK: nvgpu.device_async_wait %[[G]]
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// -----

// Check that pattern works with vector.load/vector.store.
builtin.module {
  // CHECK-LABEL: @copies_to_asyncs_load_store
  func.func @copies_to_asyncs_load_store(%a: memref<1024x1024xf32>) {
    %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    // CHECK: %[[CP0:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 4 :
    %1 = vector.load %a[%c0, %c0] : memref<1024x1024xf32>, vector<4xf32>
    vector.store %1, %0[%c0, %c0, %c0] : memref<4x32x16xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    // CHECK-NOT: nvgpu.device_async_create_group

    // CHECK: %[[CP1:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 1 :
    %2 = vector.load %a[%c0, %c4] : memref<1024x1024xf32>, vector<1xf32>
    vector.store %2, %0[%c0, %c4, %c0] : memref<4x32x16xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: %[[G:.*]] = nvgpu.device_async_create_group %[[CP0]], %[[CP1]]
    // CHECK: nvgpu.device_async_wait %[[G]]
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// -----

// Check that pattern skips unaligned and unsupported sizes.
builtin.module {
  // CHECK-LABEL: @copies_to_asyncs_load_store
  func.func @copies_to_asyncs_load_store(%a: memref<1024x1024xf32>, %b: memref<1024x1024xf16>) {
    %alloc = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
    %alloc_1 = memref.alloc() : memref<4x32x16xf16, #gpu.address_space<workgroup>>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 0.000000e+00 : f32

    // Requires 1-D vector load
    // CHECK-NOT: nvgpu.device_async_copy
    //     CHECK: vector.load
    //     CHECK: vector.store
    %1 = vector.load %a[%c0, %c4] : memref<1024x1024xf32>, vector<2x2xf32>
    vector.store %1, %alloc[%c0, %c4, %c0] : memref<4x32x16xf32, #gpu.address_space<workgroup>>, vector<2x2xf32>
    // CHECK-NOT: nvgpu.device_async_create_group

    // CHECK-NOT: nvgpu.device_async_copy
    //     CHECK: vector.load
    //     CHECK: vector.store
    %2 = vector.load %b[%c0, %c4] : memref<1024x1024xf16>, vector<1xf16>
    vector.store %2, %alloc_1[%c0, %c4, %c0] : memref<4x32x16xf16, #gpu.address_space<workgroup>>, vector<1xf16>
    // CHECK-NOT: nvgpu.device_async_create_group
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %top_level_func : (!transform.any_op) -> (!transform.any_op)
  }
}

// -----

// vector.transfer_read with a mask.
builtin.module {
  // CHECK-LABEL: @read_with_mask(
  // CHECK-SAME: %{{.*}}: memref<1024x1024xf32>, %[[sz:.*]]: index
  func.func @read_with_mask(%a: memref<1024x1024xf32>, %sz: index) {
    %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    // CHECK: nvgpu.device_async_copy {{.*}}, {{.*}}, 4, %[[sz]] {bypassL1} :
    %mask = vector.create_mask %sz : vector<4xi1>
    %1 = vector.transfer_read %a[%c0, %c0], %cst_0, %mask {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
    vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>

    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.create_async_groups %top_level_func {bypass_l1} : (!transform.any_op) -> (!transform.any_op)
  }
}
