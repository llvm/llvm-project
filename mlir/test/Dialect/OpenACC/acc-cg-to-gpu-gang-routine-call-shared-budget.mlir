// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu{max-workgroup-shared-memory=300}))" | FileCheck %s
// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu{max-workgroup-shared-memory=2100}))" | FileCheck %s --check-prefix=WIDE

// A gang routine may place up to 256 bytes of its privates in workgroup slots
// of its own, and a vector routine 64 bytes for each of 32 ThreadY rows, so a
// kernel calling one reserves that much of its shared memory budget. With a
// 300-byte budget, a 128-byte gang private fits next to neither; with 2100
// bytes, it fits next to the gang routine only. The slots of routines called
// by a routine defined in this module are reserved as well.

func.func private @gang_routine() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_gang]>}
acc.routine @acc_routine_gang func(@gang_routine) gang
func.func private @vector_routine() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_vector]>}
acc.routine @acc_routine_vector func(@vector_routine) vector
func.func @worker_routine() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_worker]>} {
  func.call @vector_routine() : () -> ()
  return
}
acc.routine @acc_routine_worker func(@worker_routine) worker

// CHECK-LABEL: func.func @gang_private_without_call
// CHECK:         gpu.launch
// CHECK:           acc.gpu_shared_memory

func.func @gang_private_without_call(%arg0: memref<32xi32>) {
  %c32_pw = arith.constant 32 : index
  %c1_pw = arith.constant 1 : index
  %par_bx = acc.par_width %c32_pw par_dim(#acc.par_dim<block_x>)
  %par_ty = acc.par_width %c1_pw par_dim(#acc.par_dim<thread_y>)
  %par_tx = acc.par_width %c32_pw par_dim(#acc.par_dim<thread_x>)
  acc.kernel_environment {
    %priv = acc.privatize : () -> !acc.private_type<memref<32xi32>>
    acc.compute_region launch(%grid = %par_bx, %worker = %par_ty, %block = %par_tx)
        ins(%arr = %arg0, %arg10 = %priv)
        : (memref<32xi32>, !acc.private_type<memref<32xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      // Outer gang-redundant loop: sequential over all blocks.
      scf.parallel (%ni) = (%c0) to (%c4) step (%c1) {
        // Partitioned gang: block_x wrapping a sequential grid-stride remainder.
        scf.parallel (%g) = (%c0) to (%grid) step (%c1) {
          scf.parallel (%grem) = (%g) to (%c32) step (%grid) {
            %work = acc.private_local %arg10
                : (!acc.private_type<memref<32xi32>>) -> memref<32xi32>
            scf.parallel (%v) = (%c0) to (%block) step (%c1) {
              %ld = memref.load %arr[%v] : memref<32xi32>
              memref.store %ld, %work[%v] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.parallel (%v2) = (%c0) to (%block) step (%c1) {
              %val = memref.load %work[%v2] : memref<32xi32>
              memref.store %val, %arr[%v2] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[block_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>,
         acc.gpu_block_redundant = #acc.gpu_block_redundant}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  return
}

// CHECK-LABEL: func.func @gang_private_with_gang_routine_call
// CHECK:         gpu.launch
// CHECK-NOT:       acc.gpu_shared_memory
// CHECK:           func.call @gang_routine
// CHECK-NOT:       acc.gpu_shared_memory
// CHECK:         return

// WIDE-LABEL: func.func @gang_private_with_gang_routine_call
// WIDE:         gpu.launch
// WIDE:           acc.gpu_shared_memory

func.func @gang_private_with_gang_routine_call(%arg0: memref<32xi32>) {
  %c32_pw = arith.constant 32 : index
  %c1_pw = arith.constant 1 : index
  %par_bx = acc.par_width %c32_pw par_dim(#acc.par_dim<block_x>)
  %par_ty = acc.par_width %c1_pw par_dim(#acc.par_dim<thread_y>)
  %par_tx = acc.par_width %c32_pw par_dim(#acc.par_dim<thread_x>)
  acc.kernel_environment {
    %priv = acc.privatize : () -> !acc.private_type<memref<32xi32>>
    acc.compute_region launch(%grid = %par_bx, %worker = %par_ty, %block = %par_tx)
        ins(%arr = %arg0, %arg10 = %priv)
        : (memref<32xi32>, !acc.private_type<memref<32xi32>>) {
      func.call @gang_routine() : () -> ()
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      // Outer gang-redundant loop: sequential over all blocks.
      scf.parallel (%ni) = (%c0) to (%c4) step (%c1) {
        // Partitioned gang: block_x wrapping a sequential grid-stride remainder.
        scf.parallel (%g) = (%c0) to (%grid) step (%c1) {
          scf.parallel (%grem) = (%g) to (%c32) step (%grid) {
            %work = acc.private_local %arg10
                : (!acc.private_type<memref<32xi32>>) -> memref<32xi32>
            scf.parallel (%v) = (%c0) to (%block) step (%c1) {
              %ld = memref.load %arr[%v] : memref<32xi32>
              memref.store %ld, %work[%v] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.parallel (%v2) = (%c0) to (%block) step (%c1) {
              %val = memref.load %work[%v2] : memref<32xi32>
              memref.store %val, %arr[%v2] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[block_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>,
         acc.gpu_block_redundant = #acc.gpu_block_redundant}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  return
}

// CHECK-LABEL: func.func @gang_private_with_vector_routine_call
// CHECK-NOT:       acc.gpu_shared_memory
// CHECK:         return

// WIDE-LABEL: func.func @gang_private_with_vector_routine_call
// WIDE:         gpu.launch
// WIDE-NOT:       acc.gpu_shared_memory
// WIDE:           func.call @vector_routine
// WIDE-NOT:       acc.gpu_shared_memory
// WIDE:         return

func.func @gang_private_with_vector_routine_call(%arg0: memref<32xi32>) {
  %c32_pw = arith.constant 32 : index
  %c1_pw = arith.constant 1 : index
  %par_bx = acc.par_width %c32_pw par_dim(#acc.par_dim<block_x>)
  %par_ty = acc.par_width %c1_pw par_dim(#acc.par_dim<thread_y>)
  %par_tx = acc.par_width %c32_pw par_dim(#acc.par_dim<thread_x>)
  acc.kernel_environment {
    %priv = acc.privatize : () -> !acc.private_type<memref<32xi32>>
    acc.compute_region launch(%grid = %par_bx, %worker = %par_ty, %block = %par_tx)
        ins(%arr = %arg0, %arg10 = %priv)
        : (memref<32xi32>, !acc.private_type<memref<32xi32>>) {
      func.call @vector_routine() : () -> ()
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      // Outer gang-redundant loop: sequential over all blocks.
      scf.parallel (%ni) = (%c0) to (%c4) step (%c1) {
        // Partitioned gang: block_x wrapping a sequential grid-stride remainder.
        scf.parallel (%g) = (%c0) to (%grid) step (%c1) {
          scf.parallel (%grem) = (%g) to (%c32) step (%grid) {
            %work = acc.private_local %arg10
                : (!acc.private_type<memref<32xi32>>) -> memref<32xi32>
            scf.parallel (%v) = (%c0) to (%block) step (%c1) {
              %ld = memref.load %arr[%v] : memref<32xi32>
              memref.store %ld, %work[%v] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.parallel (%v2) = (%c0) to (%block) step (%c1) {
              %val = memref.load %work[%v2] : memref<32xi32>
              memref.store %val, %arr[%v2] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[block_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>,
         acc.gpu_block_redundant = #acc.gpu_block_redundant}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  return
}

// WIDE-LABEL: func.func @gang_private_with_worker_routine_call
// WIDE:         gpu.launch
// WIDE-NOT:       acc.gpu_shared_memory
// WIDE:           func.call @worker_routine
// WIDE-NOT:       acc.gpu_shared_memory
// WIDE:         return

func.func @gang_private_with_worker_routine_call(%arg0: memref<32xi32>) {
  %c32_pw = arith.constant 32 : index
  %c1_pw = arith.constant 1 : index
  %par_bx = acc.par_width %c32_pw par_dim(#acc.par_dim<block_x>)
  %par_ty = acc.par_width %c1_pw par_dim(#acc.par_dim<thread_y>)
  %par_tx = acc.par_width %c32_pw par_dim(#acc.par_dim<thread_x>)
  acc.kernel_environment {
    %priv = acc.privatize : () -> !acc.private_type<memref<32xi32>>
    acc.compute_region launch(%grid = %par_bx, %worker = %par_ty, %block = %par_tx)
        ins(%arr = %arg0, %arg10 = %priv)
        : (memref<32xi32>, !acc.private_type<memref<32xi32>>) {
      func.call @worker_routine() : () -> ()
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      // Outer gang-redundant loop: sequential over all blocks.
      scf.parallel (%ni) = (%c0) to (%c4) step (%c1) {
        // Partitioned gang: block_x wrapping a sequential grid-stride remainder.
        scf.parallel (%g) = (%c0) to (%grid) step (%c1) {
          scf.parallel (%grem) = (%g) to (%c32) step (%grid) {
            %work = acc.private_local %arg10
                : (!acc.private_type<memref<32xi32>>) -> memref<32xi32>
            scf.parallel (%v) = (%c0) to (%block) step (%c1) {
              %ld = memref.load %arr[%v] : memref<32xi32>
              memref.store %ld, %work[%v] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.parallel (%v2) = (%c0) to (%block) step (%c1) {
              %val = memref.load %work[%v2] : memref<32xi32>
              memref.store %val, %arr[%v2] : memref<32xi32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[block_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>,
         acc.gpu_block_redundant = #acc.gpu_block_redundant}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  return
}
