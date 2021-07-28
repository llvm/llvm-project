// RUN: mlir-opt %s -convert-vector-to-llvm="enable-arm-sve" | mlir-opt | FileCheck %s

// CHECK: memcopy([[SRC:%arg[0-9]+]]: memref<?xf32>, [[DST:%arg[0-9]+]]
func @memcopy(%src : memref<?xf32>, %dst : memref<?xf32>, %size : index) {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %vs = arm_sve.vector_scale : index
  %step = muli %c4, %vs : index

  // CHECK: scf.for [[LOOPIDX:%arg[0-9]+]] = {{.*}}
  scf.for %i0 = %c0 to %size step %step {
    // CHECK: [[SRCMRS:%[0-9]+]] = builtin.unrealized_conversion_cast [[SRC]] : memref<?xf32> to !llvm.struct<(ptr<f32>
    // CHECK: [[SRCIDX:%[0-9]+]] = builtin.unrealized_conversion_cast [[LOOPIDX]] : index to i64
    // CHECK: [[SRCMEM:%[0-9]+]] = llvm.extractvalue [[SRCMRS]][1] : !llvm.struct<(ptr<f32>
    // CHECK-NEXT: [[SRCPTR:%[0-9]+]] = llvm.getelementptr [[SRCMEM]]{{.}}[[SRCIDX]]{{.}} : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    // CHECK-NEXT: [[SRCVPTR:%[0-9]+]] = llvm.bitcast [[SRCPTR]] : !llvm.ptr<f32> to !llvm.ptr<vec<? x 4 x f32>>
    // CHECK-NEXT: [[LDVAL:%[0-9]+]] = llvm.load [[SRCVPTR]] : !llvm.ptr<vec<? x 4 x f32>>
    %0 = arm_sve.load %src[%i0] : !arm_sve.vector<4xf32> from memref<?xf32>
    // CHECK: [[DSTMRS:%[0-9]+]] = builtin.unrealized_conversion_cast [[DST]] : memref<?xf32> to !llvm.struct<(ptr<f32>
    // CHECK: [[DSTIDX:%[0-9]+]] = builtin.unrealized_conversion_cast [[LOOPIDX]] : index to i64
    // CHECK: [[DSTMEM:%[0-9]+]] = llvm.extractvalue [[DSTMRS]][1] : !llvm.struct<(ptr<f32>
    // CHECK-NEXT: [[DSTPTR:%[0-9]+]] = llvm.getelementptr [[DSTMEM]]{{.}}[[DSTIDX]]{{.}} : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    // CHECK-NEXT: [[DSTVPTR:%[0-9]+]] = llvm.bitcast [[DSTPTR]] : !llvm.ptr<f32> to !llvm.ptr<vec<? x 4 x f32>>
    // CHECK-NEXT: llvm.store [[LDVAL]], [[DSTVPTR]] : !llvm.ptr<vec<? x 4 x f32>>
    arm_sve.store %0, %dst[%i0] : !arm_sve.vector<4xf32> to memref<?xf32>
  }

  return
}
