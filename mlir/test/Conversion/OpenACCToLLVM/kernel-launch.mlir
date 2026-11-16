// RUN: mlir-opt %s -acc-to-llvm -split-input-file | FileCheck %s

// A parallel construct with a copy clause and a scalar that has no clause. The
// data region of the construct is its implicit data region, so its entry and
// exit calls carry that region flag. The entry call returns the device address,
// which the kernel receives as a device pointer. The scalar is passed by value.
// A kernel that the module does not define is declared, and the runtime
// identifies it by the address of that declaration.
// copy = to|from|return_param = 0x43 = 67.
// devptr|literal = 0x500 = 1280.
// literal = 0x100 = 256.

// CHECK-LABEL: llvm.func @loop_0()
// CHECK: llvm.func @parallel_copy
// CHECK: llvm.mlir.addressof @acc.ident
// CHECK: %[[IMPLICIT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: llvm.mlir.constant(67 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin(%{{[^,]+}}, %[[IMPLICIT]],
// CHECK: %[[A_SLOT:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
// CHECK: %[[A_DEVICE:.*]] = llvm.load %[[A_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[KERNEL:.*]] = llvm.mlir.addressof @loop_0 : !llvm.ptr
// CHECK: llvm.mlir.constant(1280 : i64) : i64
// CHECK: llvm.mlir.constant(256 : i64) : i64
// CHECK: llvm.call @__tgt_acc_kernel(%{{[^,]+}}, %[[KERNEL]],
// CHECK: llvm.call @abort
// CHECK: %[[IMPLICIT_END:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_end(%{{[^,]+}}, %[[IMPLICIT_END]],

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @loop_0(%a: !llvm.ptr, %n: i64) kernel {
      gpu.return
    }
  }

  func.func @parallel_copy(%a: !llvm.ptr, %n: i64) {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %copy = acc.copyin varPtr(%a : !llvm.ptr) varType(f32)
        dataClause(acc_copy) name("a") -> !llvm.ptr
    acc.kernel_environment dataOperands(%copy : !llvm.ptr) {
      gpu.launch_func @device_module::@loop_0
          blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1)
          args(%copy : !llvm.ptr, %n : i64)
    }
    acc.copyout accPtr(%copy : !llvm.ptr) to varPtr(%a : !llvm.ptr)
        varType(f32) dataClause(acc_copy) name("a")
    return
  }
}

// -----

// A deviceptr clause names an address that is already on the device. It is
// mapped as devptr = 0x400 = 1024, and no device address is read back for it.

// CHECK-LABEL: llvm.func @parallel_deviceptr
// CHECK-NOT: llvm.mlir.constant(64 : i64)
// CHECK: %[[DEVPTR:.*]] = llvm.mlir.constant(1024 : i64) : i64
// CHECK: llvm.store %[[DEVPTR]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK-NOT: llvm.mlir.constant(64 : i64)
// CHECK: llvm.call @__tgt_acc_kernel
// CHECK: llvm.call @__tgt_acc_data_end

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @region_0(%p: !llvm.ptr) kernel {
      gpu.return
    }
  }

  func.func @parallel_deviceptr(%p: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %dev = acc.deviceptr varPtr(%p : !llvm.ptr) varType(f32) name("p")
        -> !llvm.ptr
    acc.kernel_environment dataOperands(%dev : !llvm.ptr) {
      gpu.launch_func @device_module::@region_0
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%dev : !llvm.ptr)
    }
    return
  }
}

// -----

// A present clause maps its operand with present = 0x100000 = 1048576, and its
// device address is read back through return_param = 0x40 = 64. The kernel
// receives that address.

// CHECK-LABEL: llvm.func @parallel_present
// CHECK: %[[PRESENT:.*]] = llvm.mlir.constant(1048640 : i64) : i64
// CHECK: llvm.store %[[PRESENT]], %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.mlir.constant(1280 : i64) : i64
// CHECK: llvm.call @__tgt_acc_kernel

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @region_0(%a: !llvm.ptr) kernel {
      gpu.return
    }
  }

  func.func @parallel_present(%a: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %present = acc.present varPtr(%a : !llvm.ptr) varType(f32) name("a")
        -> !llvm.ptr
    acc.kernel_environment dataOperands(%present : !llvm.ptr) {
      gpu.launch_func @device_module::@region_0
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%present : !llvm.ptr)
    }
    return
  }
}

// -----

// A construct with async and wait clauses and no data clauses. The wait
// precedes the launch, and the async queue of the construct is the queue of the
// kernel call. The launch geometry fills the argument struct. The vector length
// is the first block dimension, and the number of workers is the product of the
// other two. Any status other than success reaches the host fallback.
// literal = 0x100 = 256.

// CHECK-LABEL: llvm.func @parallel_async_wait
// CHECK-SAME: (%[[X:.*]]: i32, %[[SMEM:.*]]: i32, %[[ASYNC:.*]]: i32, %[[WAIT:.*]]: i32)
// CHECK-NOT: llvm.call @__tgt_acc_data_begin
// CHECK: %[[GRID_X:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[GRID_Y:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK: %[[GRID_Z:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK: %[[BLOCK_X:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: %[[BLOCK_Y:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK: %[[BLOCK_Z:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK: %[[QUEUE:.*]] = llvm.sext %[[ASYNC]] : i32 to i64
// CHECK: llvm.call @__tgt_acc_wait
// CHECK: %[[KERNEL:.*]] = llvm.mlir.addressof @region_0 : !llvm.ptr
// CHECK: llvm.store %[[X]], %{{.*}} : i32, !llvm.ptr
// CHECK: llvm.mlir.constant(256 : i64) : i64
// CHECK: %[[WORKERS:.*]] = llvm.mul %[[BLOCK_Y]], %[[BLOCK_Z]] : i64
// CHECK: llvm.insertvalue %[[WORKERS]], %{{.*}}[11]
// CHECK: llvm.insertvalue %[[BLOCK_X]], %{{.*}}[12]
// CHECK: %[[SMEM64:.*]] = llvm.sext %[[SMEM]] : i32 to i64
// CHECK: llvm.insertvalue %[[SMEM64]], %{{.*}}[13]
// CHECK: llvm.insertvalue %[[GRID_X]], %{{.*}}[10, 0]
// CHECK: llvm.insertvalue %[[GRID_Y]], %{{.*}}[10, 1]
// CHECK: llvm.insertvalue %[[GRID_Z]], %{{.*}}[10, 2]
// CHECK: llvm.mlir.addressof @acc.kernel_name.region_0
// CHECK: llvm.mlir.addressof @acc.ident
// CHECK: %[[KERNEL_FLAGS:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[STATUS:.*]] = llvm.call @__tgt_acc_kernel(%{{[^,]+}}, %[[KERNEL]], %[[KERNEL_FLAGS]], %{{[^,]+}}, %{{[^,]+}}, %[[QUEUE]], %{{[^,]+}}, %{{[^,]+}})
// CHECK: %[[SUCCESS:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[FAILED:.*]] = llvm.icmp "ne" %[[STATUS]], %[[SUCCESS]] : i32
// CHECK: llvm.cond_br %[[FAILED]]
// CHECK-NOT: llvm.call @__tgt_acc_data_end

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @region_0(%x: i32) kernel {
      gpu.return
    }
  }

  func.func @parallel_async_wait(%x: i32, %smem: i32, %async: i32, %wait: i32) {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c8 = arith.constant 8 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    acc.kernel_environment async(%async : i32) wait(%wait : i32) {
      gpu.launch_func @device_module::@region_0
          blocks in (%c2, %c3, %c5) threads in (%c8, %c6, %c7)
          dynamic_shared_memory_size %smem args(%x : i32)
    }
    return
  }
}

// -----

// The runtime identifies a kernel by the address of a host function that has
// the kernel's name. The call takes the address of that function when the
// module defines it, so no declaration is added.

// CHECK-LABEL: llvm.func internal @region_0()
// CHECK-NOT: llvm.func @region_0
// CHECK-LABEL: llvm.func @parallel_with_host_stub
// CHECK: %[[KERNEL:.*]] = llvm.mlir.addressof @region_0 : !llvm.ptr
// CHECK: llvm.call @__tgt_acc_kernel(%{{[^,]+}}, %[[KERNEL]],

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @region_0(%a: !llvm.ptr) kernel {
      gpu.return
    }
  }

  llvm.func internal @region_0() {
    llvm.return
  }

  func.func @parallel_with_host_stub(%a: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    acc.kernel_environment {
      gpu.launch_func @device_module::@region_0
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%a : !llvm.ptr)
    }
    return
  }
}

// -----

// A memref operand is mapped as its descriptor fields, and only the first
// field, the one holding the allocated pointer, asks the runtime to return the
// device address. copy = to|from|return_param = 0x43 = 67 and to|from =
// 0x3 = 3.

// CHECK-LABEL: llvm.func @parallel_copy_memref
// CHECK: llvm.mlir.constant(67 : i64) : i64
// CHECK: llvm.mlir.constant(3 : i64) : i64
// CHECK: llvm.mlir.constant(3 : i64) : i64
// CHECK: llvm.mlir.constant(3 : i64) : i64
// CHECK: llvm.mlir.constant(3 : i64) : i64
// CHECK-NOT: llvm.mlir.constant(67 : i64)
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.call @__tgt_acc_kernel

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @region_0(%a: memref<100xf32>) kernel {
      gpu.return
    }
  }

  func.func @parallel_copy_memref(%a: memref<100xf32>) {
    %c1 = arith.constant 1 : index
    %copy = acc.copyin varPtr(%a : memref<100xf32>) varType(f32)
        dataClause(acc_copy) name("a") -> memref<100xf32>
    acc.kernel_environment dataOperands(%copy : memref<100xf32>) {
      gpu.launch_func @device_module::@region_0
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%copy : memref<100xf32>)
    }
    acc.copyout accPtr(%copy : memref<100xf32>) to varPtr(%a : memref<100xf32>)
        varType(f32) dataClause(acc_copy) name("a")
    return
  }
}

// -----

// A memref that no clause maps is described to the runtime like a mapped one.
// The function parameter is converted to a descriptor, so the element size
// comes from the memref type.

// CHECK-LABEL: llvm.func @parallel_unmapped_memref
// CHECK: llvm.mlir.constant(1 : i8) : i8
// CHECK: llvm.alloca %{{.*}} x !llvm.struct<(i32, i8, i64, ptr)>
// CHECK: llvm.call @__tgt_acc_kernel

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @region_0(%m: memref<4xf32>) kernel {
      gpu.return
    }
  }

  func.func @parallel_unmapped_memref(%m: memref<4xf32>) {
    %c1 = arith.constant 1 : index
    acc.kernel_environment {
      gpu.launch_func @device_module::@region_0
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%m : memref<4xf32>)
    }
    return
  }
}

// -----

// An object of a declare device_resident clause is mapped by the declare
// directive outside of the environment, which has no data operand for it. The
// kernel looks up that mapping with present = 0x100000 = 1048576, and not with
// the device_resident = 0x40000 = 262144 of the declaration.

// CHECK-LABEL: llvm.func @parallel_declare_device_resident
// CHECK: llvm.mlir.constant(262144 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK-NOT: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.mlir.constant(1048576 : i64) : i64
// CHECK: llvm.call @__tgt_acc_kernel
// CHECK: llvm.call @__tgt_acc_data_end

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @region_0(%b: !llvm.ptr) kernel {
      gpu.return
    }
  }

  func.func @parallel_declare_device_resident(%b: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %resident = acc.declare_device_resident varPtr(%b : !llvm.ptr)
        varType(f32) name("b") -> !llvm.ptr
    %token = acc.declare_enter dataOperands(%resident : !llvm.ptr)
    acc.kernel_environment {
      gpu.launch_func @device_module::@region_0
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%resident : !llvm.ptr)
    }
    acc.declare_exit token(%token) dataOperands(%resident : !llvm.ptr)
    acc.delete accPtr(%resident : !llvm.ptr)
        dataClause(acc_declare_device_resident) name("b")
    return
  }
}

// -----

// A gang private variable is a data entry in the environment region, since the
// storage is created for the kernel. The environment has no data operand for
// it, so the kernel call maps it, with private|gang_private =
// 0x4080 = 16512. The copied array is still mapped by the environment.

// CHECK-LABEL: llvm.func @parallel_gang_private
// CHECK: llvm.mlir.constant(67 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin
// CHECK-NOT: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.mlir.constant(1280 : i64) : i64
// CHECK: llvm.mlir.constant(16512 : i64) : i64
// CHECK: llvm.call @__tgt_acc_kernel
// CHECK: llvm.call @__tgt_acc_data_end

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @region_0(%a: !llvm.ptr,
                       %p: !acc.private_type<memref<99xi32>>) kernel {
      gpu.return
    }
  }

  func.func @parallel_gang_private(%a: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %size = arith.constant 396 : i64
    %copy = acc.copyin varPtr(%a : !llvm.ptr) varType(f32)
        dataClause(acc_copy) name("a") -> !llvm.ptr
    acc.kernel_environment dataOperands(%copy : !llvm.ptr) {
      %priv = acc.privatize par_dims(#acc<par_dims[block_x]>)
          : () -> !acc.private_type<memref<99xi32>>
      %map = acc.map_info varPtr(%priv : !acc.private_type<memref<99xi32>>)
          size(%size : i64) elementSize(4) descKind(none)
          mapFlags(private, gang_private) -> !acc.private_type<memref<99xi32>>
      gpu.launch_func @device_module::@region_0
          blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
          args(%copy : !llvm.ptr, %map : !acc.private_type<memref<99xi32>>)
    }
    acc.copyout accPtr(%copy : !llvm.ptr) to varPtr(%a : !llvm.ptr)
        varType(f32) dataClause(acc_copy) name("a")
    return
  }
}

// -----

// Several launches in one kernel environment share its data region, as the
// kernels of a split kernels construct would. The array a has an explicit copy
// clause and the array b is copied implicitly. The entry call returns both
// device addresses, and every kernel receives them as device pointers.
// copy = to|from|return_param = 0x43 = 67.
// implicit copy = to|from|implicit|return_param = 0x243 = 579.
// devptr|literal = 0x500 = 1280, and devptr|literal|implicit = 0x700 = 1792.
// literal = 0x100 = 256.

// CHECK-LABEL: llvm.func @kernels_with_copy
// CHECK: llvm.mlir.addressof @acc.ident
// CHECK: %[[IMPLICIT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: llvm.mlir.constant(67 : i64) : i64
// CHECK: llvm.mlir.constant(579 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_begin(%{{[^,]+}}, %[[IMPLICIT]],
// CHECK: %[[A_SLOT:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
// CHECK: %[[A_DEVICE:.*]] = llvm.load %[[A_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[B_SLOT:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
// CHECK: %[[B_DEVICE:.*]] = llvm.load %[[B_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.mlir.constant(1280 : i64) : i64
// CHECK: llvm.mlir.constant(1792 : i64) : i64
// CHECK: llvm.mlir.constant(256 : i64) : i64
// CHECK: llvm.call @__tgt_acc_kernel
// CHECK: llvm.call @abort
// CHECK-NOT: llvm.call @__tgt_acc_data_begin
// CHECK: llvm.mlir.constant(1280 : i64) : i64
// CHECK: llvm.mlir.constant(1792 : i64) : i64
// CHECK: llvm.mlir.constant(256 : i64) : i64
// CHECK: llvm.call @__tgt_acc_kernel
// CHECK: llvm.call @abort
// CHECK: %[[IMPLICIT_END:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: llvm.call @__tgt_acc_data_end(%{{[^,]+}}, %[[IMPLICIT_END]],

module attributes {gpu.container_module} {
  gpu.module @device_module {
    gpu.func @loop_0(%a: !llvm.ptr, %b: !llvm.ptr, %n: i64) kernel {
      gpu.return
    }
    gpu.func @loop_1(%a: !llvm.ptr, %b: !llvm.ptr, %n: i64) kernel {
      gpu.return
    }
  }

  func.func @kernels_with_copy(%a: !llvm.ptr, %b: !llvm.ptr, %n: i64) {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %copy_a = acc.copyin varPtr(%a : !llvm.ptr) varType(f32)
        dataClause(acc_copy) name("a") -> !llvm.ptr
    %copy_b = acc.copyin varPtr(%b : !llvm.ptr) varType(f32)
        dataClause(acc_copy) implicit(true) name("b") -> !llvm.ptr
    acc.kernel_environment dataOperands(%copy_a, %copy_b : !llvm.ptr, !llvm.ptr) {
      gpu.launch_func @device_module::@loop_0
          blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1)
          args(%copy_a : !llvm.ptr, %copy_b : !llvm.ptr, %n : i64)
      gpu.launch_func @device_module::@loop_1
          blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1)
          args(%copy_a : !llvm.ptr, %copy_b : !llvm.ptr, %n : i64)
    }
    acc.copyout accPtr(%copy_a : !llvm.ptr) to varPtr(%a : !llvm.ptr)
        varType(f32) dataClause(acc_copy) name("a")
    acc.copyout accPtr(%copy_b : !llvm.ptr) to varPtr(%b : !llvm.ptr)
        varType(f32) dataClause(acc_copy) implicit(true) name("b")
    return
  }
}
