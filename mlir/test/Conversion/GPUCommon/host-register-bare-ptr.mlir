// RUN: mlir-opt %s --gpu-to-llvm="use-bare-pointers-for-host=1" -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=BARE

module attributes {gpu.container_module} {
  func.func @host_register(%arg0: memref<4x6xf16>) {
    gpu.host_register %arg0 : memref<4x6xf16>
    gpu.host_unregister %arg0 : memref<4x6xf16>
    return
  }
}

// BARE-LABEL: llvm.func @host_register
// BARE-SAME: ({{.*}}: !llvm.ptr) {
// BARE: %[[DESC0:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BARE: %[[DESC1:.+]] = llvm.insertvalue %arg0, %[[DESC0]][0]
// BARE: %[[DESC2:.+]] = llvm.insertvalue %arg0, %[[DESC1]][1]
// BARE: %[[OFF:.+]] = llvm.mlir.constant(0 : {{.*}}) : i64
// BARE: %[[DESC3:.+]] = llvm.insertvalue %[[OFF]], %[[DESC2]][2]
// BARE: %[[SIZE0:.+]] = llvm.mlir.constant(4 : {{.*}}) : i64
// BARE: %[[DESC4:.+]] = llvm.insertvalue %[[SIZE0]], %[[DESC3]][3, 0]
// BARE: %[[STRIDE0:.+]] = llvm.mlir.constant(6 : {{.*}}) : i64
// BARE: %[[DESC5:.+]] = llvm.insertvalue %[[STRIDE0]], %[[DESC4]][4, 0]
// BARE: %[[SIZE1:.+]] = llvm.mlir.constant(6 : {{.*}}) : i64
// BARE: %[[DESC6:.+]] = llvm.insertvalue %[[SIZE1]], %[[DESC5]][3, 1]
// BARE: %[[STRIDE1:.+]] = llvm.mlir.constant(1 : {{.*}}) : i64
// BARE: %[[DESC7:.+]] = llvm.insertvalue %[[STRIDE1]], %[[DESC6]][4, 1]
// BARE: %[[RANK:.+]] = llvm.mlir.constant(2 : {{.*}}) : i64
// BARE: %[[ALLOCA:.+]] = llvm.alloca %{{.*}} x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BARE: llvm.store %[[DESC7]], %[[ALLOCA]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
// BARE: %[[NULL:.+]] = llvm.mlir.zero : !llvm.ptr
// BARE: %[[GEP:.+]] = llvm.getelementptr %[[NULL]][1] : (!llvm.ptr) -> !llvm.ptr, f16
// BARE: %[[ELTSZ:.+]] = llvm.ptrtoint %[[GEP]] : !llvm.ptr to i64
// BARE: llvm.call @mgpuMemHostRegisterMemRef(%[[RANK]], %[[ALLOCA]], %[[ELTSZ]])
// BARE: llvm.call @mgpuMemHostUnregisterMemRef(%{{.*}}, %{{.*}}, %{{.*}})

// -----

module attributes {gpu.container_module} {
  func.func @dynamic(%n: index) {
    %buf = memref.alloc(%n) : memref<?xf32>
    // expected-error @+1 {{cannot lower memref with bare pointer calling convention}}
    gpu.host_register %buf : memref<?xf32>
    return
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @unranked(%arg0: memref<*xf32>) {
    // expected-error @+1 {{custom op 'gpu.host_register' invalid kind of type specified: expected builtin.memref, but found 'memref<*xf32>'}}
    gpu.host_register %arg0 : memref<*xf32>
    return
  }
}
