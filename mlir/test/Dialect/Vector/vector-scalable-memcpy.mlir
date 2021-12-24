// RUN: mlir-opt %s -convert-vector-to-llvm | mlir-opt | FileCheck %s

// CHECK: vector_scalable_memcopy([[SRC:%arg[0-9]+]]: memref<?xf32>, [[DST:%arg[0-9]+]]
func @vector_scalable_memcopy(%src : memref<?xf32>, %dst : memref<?xf32>, %size : index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %vs = vector.vscale
  %step = arith.muli %c4, %vs : index
  // CHECK: [[SRCMRS:%[0-9]+]] = builtin.unrealized_conversion_cast [[SRC]] : memref<?xf32> to !llvm.struct<(ptr<f32>
  // CHECK: [[DSTMRS:%[0-9]+]] = builtin.unrealized_conversion_cast [[DST]] : memref<?xf32> to !llvm.struct<(ptr<f32>
  // CHECK: scf.for [[LOOPIDX:%arg[0-9]+]] = {{.*}}
  scf.for %i0 = %c0 to %size step %step {
    // CHECK: [[DATAIDX:%[0-9]+]] = builtin.unrealized_conversion_cast [[LOOPIDX]] : index to i64
    // CHECK: [[SRCMEM:%[0-9]+]] = llvm.extractvalue [[SRCMRS]][1] : !llvm.struct<(ptr<f32>
    // CHECK-NEXT: [[SRCPTR:%[0-9]+]] = llvm.getelementptr [[SRCMEM]]{{.}}[[DATAIDX]]{{.}} : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    // CHECK-NEXT: [[SRCVPTR:%[0-9]+]] = llvm.bitcast [[SRCPTR]] : !llvm.ptr<f32> to !llvm.ptr<vector<[4]xf32>>
    // CHECK-NEXT: [[LDVAL:%[0-9]+]] = llvm.load [[SRCVPTR]]{{.*}}: !llvm.ptr<vector<[4]xf32>>
    %0 = vector.load %src[%i0] : memref<?xf32>, vector<[4]xf32>
    // CHECK: [[DSTMEM:%[0-9]+]] = llvm.extractvalue [[DSTMRS]][1] : !llvm.struct<(ptr<f32>
    // CHECK-NEXT: [[DSTPTR:%[0-9]+]] = llvm.getelementptr [[DSTMEM]]{{.}}[[DATAIDX]]{{.}} : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    // CHECK-NEXT: [[DSTVPTR:%[0-9]+]] = llvm.bitcast [[DSTPTR]] : !llvm.ptr<f32> to !llvm.ptr<vector<[4]xf32>>
    // CHECK-NEXT: llvm.store [[LDVAL]], [[DSTVPTR]]{{.*}}: !llvm.ptr<vector<[4]xf32>>
    vector.store %0, %dst[%i0] : memref<?xf32>, vector<[4]xf32>
  }

  return
}
