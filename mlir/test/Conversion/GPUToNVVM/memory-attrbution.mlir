// RUN: mlir-opt --convert-gpu-to-nvvm --split-input-file %s | FileCheck %s

module attributes {gpu.kernel_module} {
  // CHECK-LABEL:  llvm.func @private
  gpu.func @private(%arg0: f32) private(%arg1: memref<4xf32, 5>) {
    // Allocate private memory inside the function.
    // CHECK: %[[size:.*]] = llvm.mlir.constant(4 : i64) : !llvm.i64
    // CHECK: %[[raw:.*]] = llvm.alloca %[[size]] x !llvm.float : (!llvm.i64) -> !llvm<"float*">

    // Populate the memref descriptor.
    // CHECK: %[[descr1:.*]] = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
    // CHECK: %[[descr2:.*]] = llvm.insertvalue %[[raw]], %[[descr1]][0]
    // CHECK: %[[descr3:.*]] = llvm.insertvalue %[[raw]], %[[descr2]][1]
    // CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK: %[[descr4:.*]] = llvm.insertvalue %[[c0]], %[[descr3]][2]
    // CHECK: %[[c4:.*]] = llvm.mlir.constant(4 : index) : !llvm.i64
    // CHECK: %[[descr5:.*]] = llvm.insertvalue %[[c4]], %[[descr4]][3, 0]
    // CHECK: %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK: %[[descr6:.*]] = llvm.insertvalue %[[c1]], %[[descr5]][4, 0]

    // "Store" lowering should work just as any other memref, only check that
    // we emit some core instructions.
    // CHECK: llvm.extractvalue %[[descr6:.*]]
    // CHECK: llvm.getelementptr
    // CHECK: llvm.store
    %c0 = constant 0 : index
    store %arg0, %arg1[%c0] : memref<4xf32, 5>

    "terminator"() : () -> ()
  }
}

// -----

module attributes {gpu.kernel_module} {
  // Workgroup buffers are allocated as globals.
  // CHECK: llvm.mlir.global internal @[[buffer:.*]]()
  // CHECK-SAME:  addr_space = 3
  // CHECK-SAME:  !llvm<"[4 x float]">

  // CHECK-LABEL: llvm.func @workgroup
  // CHECK-SAME: {
  gpu.func @workgroup(%arg0: f32) workgroup(%arg1: memref<4xf32, 3>) {
    // Get the address of the first element in the global array.
    // CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK: %[[addr:.*]] = llvm.mlir.addressof @[[buffer]] : !llvm<"[4 x float] addrspace(3)*">
    // CHECK: %[[raw:.*]] = llvm.getelementptr %[[addr]][%[[c0]], %[[c0]]]
    // CHECK-SAME: !llvm<"float addrspace(3)*">

    // Populate the memref descriptor.
    // CHECK: %[[descr1:.*]] = llvm.mlir.undef : !llvm<"{ float addrspace(3)*, float addrspace(3)*, i64, [1 x i64], [1 x i64] }">
    // CHECK: %[[descr2:.*]] = llvm.insertvalue %[[raw]], %[[descr1]][0]
    // CHECK: %[[descr3:.*]] = llvm.insertvalue %[[raw]], %[[descr2]][1]
    // CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK: %[[descr4:.*]] = llvm.insertvalue %[[c0]], %[[descr3]][2]
    // CHECK: %[[c4:.*]] = llvm.mlir.constant(4 : index) : !llvm.i64
    // CHECK: %[[descr5:.*]] = llvm.insertvalue %[[c4]], %[[descr4]][3, 0]
    // CHECK: %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK: %[[descr6:.*]] = llvm.insertvalue %[[c1]], %[[descr5]][4, 0]

    // "Store" lowering should work just as any other memref, only check that
    // we emit some core instructions.
    // CHECK: llvm.extractvalue %[[descr6:.*]]
    // CHECK: llvm.getelementptr
    // CHECK: llvm.store
    %c0 = constant 0 : index
    store %arg0, %arg1[%c0] : memref<4xf32, 3>

    "terminator"() : () -> ()
  }
}

// -----

module attributes {gpu.kernel_module} {
  // Check that the total size was computed correctly.
  // CHECK: llvm.mlir.global internal @[[buffer:.*]]()
  // CHECK-SAME:  addr_space = 3
  // CHECK-SAME:  !llvm<"[48 x float]">

  // CHECK-LABEL: llvm.func @workgroup3d
  gpu.func @workgroup3d(%arg0: f32) workgroup(%arg1: memref<4x2x6xf32, 3>) {
    // Get the address of the first element in the global array.
    // CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK: %[[addr:.*]] = llvm.mlir.addressof @[[buffer]] : !llvm<"[48 x float] addrspace(3)*">
    // CHECK: %[[raw:.*]] = llvm.getelementptr %[[addr]][%[[c0]], %[[c0]]]
    // CHECK-SAME: !llvm<"float addrspace(3)*">

    // Populate the memref descriptor.
    // CHECK: %[[descr1:.*]] = llvm.mlir.undef : !llvm<"{ float addrspace(3)*, float addrspace(3)*, i64, [3 x i64], [3 x i64] }">
    // CHECK: %[[descr2:.*]] = llvm.insertvalue %[[raw]], %[[descr1]][0]
    // CHECK: %[[descr3:.*]] = llvm.insertvalue %[[raw]], %[[descr2]][1]
    // CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK: %[[descr4:.*]] = llvm.insertvalue %[[c0]], %[[descr3]][2]
    // CHECK: %[[c6:.*]] = llvm.mlir.constant(6 : index) : !llvm.i64
    // CHECK: %[[descr5:.*]] = llvm.insertvalue %[[c6]], %[[descr4]][3, 2]
    // CHECK: %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
    // CHECK: %[[descr6:.*]] = llvm.insertvalue %[[c1]], %[[descr5]][4, 2]
    // CHECK: %[[c2:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
    // CHECK: %[[descr7:.*]] = llvm.insertvalue %[[c2]], %[[descr6]][3, 1]
    // CHECK: %[[c6:.*]] = llvm.mlir.constant(6 : index) : !llvm.i64
    // CHECK: %[[descr8:.*]] = llvm.insertvalue %[[c6]], %[[descr7]][4, 1]
    // CHECK: %[[c4:.*]] = llvm.mlir.constant(4 : index) : !llvm.i64
    // CHECK: %[[descr9:.*]] = llvm.insertvalue %[[c4]], %[[descr8]][3, 0]
    // CHECK: %[[c12:.*]] = llvm.mlir.constant(12 : index) : !llvm.i64
    // CHECK: %[[descr10:.*]] = llvm.insertvalue %[[c12]], %[[descr9]][4, 0]

    %c0 = constant 0 : index
    store %arg0, %arg1[%c0,%c0,%c0] : memref<4x2x6xf32, 3>
    "terminator"() : () -> ()
  }
}

// -----

module attributes {gpu.kernel_module} {
  // Check that several buffers are defined.
  // CHECK: llvm.mlir.global internal @[[buffer1:.*]]()
  // CHECK-SAME:  !llvm<"[1 x float]">
  // CHECK: llvm.mlir.global internal @[[buffer2:.*]]()
  // CHECK-SAME:  !llvm<"[2 x float]">

  // CHECK-LABEL: llvm.func @multiple
  gpu.func @multiple(%arg0: f32)
      workgroup(%arg1: memref<1xf32, 3>, %arg2: memref<2xf32, 3>)
      private(%arg3: memref<3xf32, 5>, %arg4: memref<4xf32, 5>) {

    // Workgroup buffers.
    // CHECK: llvm.mlir.addressof @[[buffer1]]
    // CHECK: llvm.mlir.addressof @[[buffer2]]

    // Private buffers.
    // CHECK: %[[c3:.*]] = llvm.mlir.constant(3 : i64)
    // CHECK: llvm.alloca %[[c3]] x !llvm.float
    // CHECK: %[[c4:.*]] = llvm.mlir.constant(4 : i64)
    // CHECK: llvm.alloca %[[c4]] x !llvm.float

    %c0 = constant 0 : index
    store %arg0, %arg1[%c0] : memref<1xf32, 3>
    store %arg0, %arg2[%c0] : memref<2xf32, 3>
    store %arg0, %arg3[%c0] : memref<3xf32, 5>
    store %arg0, %arg4[%c0] : memref<4xf32, 5>
    "terminator"() : () -> ()
  }
}
