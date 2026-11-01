// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu{device-type=nvidia}))" --split-input-file | FileCheck %s

// A loop partitioned across threads inside a sequential loop writes memory that
// other threads of the workgroup may read afterwards. A workgroup barrier must
// follow the partitioned loop, otherwise a thread can read a value that another
// warp has not yet stored.

// Only 4 of the 128 threads store (the others skip the guarded body), yet every
// thread reaches the load that follows. In the same iteration of the sequential
// loop that load reads an element stored by another thread, so the barrier must
// be reached by all threads, outside the guard.
// CHECK-LABEL: func.func @write_then_read
// CHECK:         scf.for
// CHECK:           scf.if
// CHECK-NEXT:        memref.store
// CHECK-NEXT:      }
// CHECK:           gpu.block_dim x
// CHECK:           gpu.barrier
// CHECK:           memref.load
func.func @write_then_read(%a: memref<128xf32>, %b: memref<128xf32>) {
  %c1_pw = arith.constant 1 : index
  %c128_pw = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128_pw par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a, %mb = %b) : (memref<128xf32>, memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %one = arith.constant 1.0 : f32
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.for %it = %c0 to %c4 step %c1 {
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          %active = arith.cmpi ult, %i, %c4 : index
          scf.if %active {
            memref.store %one, %ma[%i] : memref<128xf32>
          }
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        %v = memref.load %ma[%c3] : memref<128xf32>
        memref.store %v, %mb[%it] : memref<128xf32>
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}

// -----

// Nothing reads the stored memory after the partitioned loop, so no barrier is
// needed.
// CHECK-LABEL: func.func @write_only
// CHECK:         scf.for
// CHECK-NEXT:      memref.store
// CHECK-NEXT:    }
// CHECK-NOT:     gpu.barrier
func.func @write_only(%a: memref<128xf32>) {
  %c1_pw = arith.constant 1 : index
  %c128_pw = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128_pw par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a) : (memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %one = arith.constant 1.0 : f32
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.for %it = %c0 to %c4 step %c1 {
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          memref.store %one, %ma[%i] : memref<128xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}

// -----

// Double buffering: the partitioned loop is the only operation in the
// sequential loop body. It reads one column and writes the other at
// neighbouring indices, so the next iteration's loads see values stored by
// other threads and a barrier is required after the loop.
// CHECK-LABEL: func.func @double_buffer
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.store
// CHECK:           gpu.block_dim x
// CHECK:           gpu.barrier
// CHECK:           nvvm.barrier
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    }
// CHECK-NEXT:    gpu.terminator
func.func @double_buffer(%a: memref<130x2xf32>) {
  %c1_pw = arith.constant 1 : index
  %c128_pw = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128_pw par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a) : (memref<130x2xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.for %it = %c0 to %c4 step %c1 {
        %old = arith.remui %it, %c2 : index
        %new = arith.subi %c1, %old : index
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          %im = arith.addi %i, %c1 : index
          %ip = arith.addi %i, %c2 : index
          %l = memref.load %ma[%i, %old] : memref<130x2xf32>
          %r = memref.load %ma[%ip, %old] : memref<130x2xf32>
          %s = arith.addf %l, %r : f32
          memref.store %s, %ma[%im, %new] : memref<130x2xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}

// -----

// A partitioned loop that stages data through its own workgroup shared-memory
// tile synchronizes that tile itself; no additional barrier is added after it.
// CHECK-LABEL: func.func @shared_memory_tile
// CHECK:         scf.for
// CHECK:           acc.gpu_shared_memory
// CHECK:           memref.store
// CHECK:           memref.store
// CHECK-NEXT:    }
// CHECK-NOT:     gpu.barrier
func.func @shared_memory_tile(%a: memref<128xf32>) {
  %c1_pw = arith.constant 1 : index
  %c128_pw = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128_pw par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a) : (memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.for %it = %c0 to %c4 step %c1 {
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          %tile = acc.gpu_shared_memory()
              <{num_copies = 1 : i64, static_upper_bound_bytes = 512 : i64}>
              : () -> memref<128xf32, #gpu.address_space<workgroup>>
          %v = memref.load %ma[%i] : memref<128xf32>
          memref.store %v, %tile[%i] : memref<128xf32, #gpu.address_space<workgroup>>
          memref.store %v, %ma[%i] : memref<128xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}

// -----

// Write after write: two partitioned loops in the sequential loop body store
// to the same array, but a thread stores element i in the first loop and
// element 127 - i in the second. Nothing reads the array, yet without a
// barrier a thread can run ahead into the second loop and store an element
// before the thread that owns it in the first loop has stored it, which leaves
// the older value in memory.
// CHECK-LABEL: func.func @write_after_write
// CHECK:         scf.for
// CHECK:           memref.store
// CHECK:           gpu.block_dim x
// CHECK:           gpu.barrier
// CHECK:           memref.store
func.func @write_after_write(%a: memref<128xf32>) {
  %c1_pw = arith.constant 1 : index
  %c128_pw = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128_pw par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a) : (memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c127 = arith.constant 127 : index
    %c128 = arith.constant 128 : index
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.for %it = %c0 to %c4 step %c1 {
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          memref.store %one, %ma[%i] : memref<128xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.parallel (%j) = (%c0) to (%c128) step (%c1) {
          %r = arith.subi %c127, %j : index
          memref.store %two, %ma[%r] : memref<128xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}

// -----

// Write after read: the first partitioned loop only reads its neighbour's
// element and the second loop overwrites the array. Without a barrier a thread
// can overwrite element i + 1 before the thread that needs the old value has
// read it. The loaded value is unused so that only the read matters.
// CHECK-LABEL: func.func @write_after_read
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           gpu.block_dim x
// CHECK:           gpu.barrier
// CHECK:           memref.store
func.func @write_after_read(%a: memref<130xf32>) {
  %c1_pw = arith.constant 1 : index
  %c128_pw = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128_pw par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a) : (memref<130xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0.0 : f32
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.for %it = %c0 to %c4 step %c1 {
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          %n = arith.addi %i, %c1 : index
          %v = memref.load %ma[%n] : memref<130xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.parallel (%j) = (%c0) to (%c128) step (%c1) {
          memref.store %zero, %ma[%j] : memref<130xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}

// -----

// The code after the partitioned loop only touches a different array, so it
// does not conflict with the loop and no barrier is needed. The two arrays are
// separate allocations, so they cannot overlap.
// CHECK-LABEL: func.func @different_buffers
// CHECK:         scf.for
// CHECK-NOT:     gpu.barrier
// CHECK:         gpu.terminator
func.func @different_buffers() {
  %a = memref.alloc() : memref<128xf32>
  %b = memref.alloc() : memref<4xf32>
  %c1_pw = arith.constant 1 : index
  %c128_pw = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128_pw par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a, %mb = %b) : (memref<128xf32>, memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %one = arith.constant 1.0 : f32
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.for %it = %c0 to %c4 step %c1 {
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          memref.store %one, %ma[%i] : memref<128xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        %v = memref.load %mb[%it] : memref<4xf32>
        memref.store %v, %mb[%it] : memref<4xf32>
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}

// -----

// A buffer private to a gang is shared by all threads of the block. The vector
// loop that fills it and the vector loop that reads it back in reverse order
// run in a sequential loop. A barrier is needed after each of them: after the
// first so that the reads see all stores, after the second so that the next
// iteration does not overwrite elements that are still being read.
// CHECK-LABEL: func.func @gang_private_buffer
// CHECK:         scf.for
// CHECK:           memref.store
// CHECK:           nvvm.barrier
// CHECK-NOT:       {{(nvvm|gpu)\.barrier}}
// CHECK:           memref.load
// CHECK:           memref.store
// CHECK:           nvvm.barrier
// CHECK-NOT:       {{(nvvm|gpu)\.barrier}}
// CHECK:         gpu.terminator
func.func @gang_private_buffer(%a: memref<256xf32>, %out: memref<256xf32>,
                               %nb: index, %nv: index) {
  %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %nv par_dim(#acc.par_dim<thread_x>)
  %priv = acc.privatize par_dims(#acc<par_dims[block_x]>) : () -> !acc.private_type<memref<256xf32>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a, %mo = %out, %mp = %priv) : (memref<256xf32>, memref<256xf32>, !acc.private_type<memref<256xf32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c255 = arith.constant 255 : index
    %c256 = arith.constant 256 : index
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.parallel (%i) = (%g) to (%c32) step (%kbx) {
        %t = acc.private_local %mp : (!acc.private_type<memref<256xf32>>) -> memref<256xf32>
        scf.for %k = %c0 to %c4 step %c1 {
          scf.parallel (%v) = (%c0) to (%ktx) step (%c1) {
            scf.parallel (%e) = (%v) to (%c256) step (%ktx) {
              %x = memref.load %ma[%e] : memref<256xf32>
              memref.store %x, %t[%e] : memref<256xf32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[sequential]>}
            scf.reduce
          } {acc.par_dims = #acc<par_dims[thread_x]>}
          scf.parallel (%v2) = (%c0) to (%ktx) step (%c1) {
            scf.parallel (%e2) = (%v2) to (%c256) step (%ktx) {
              %r = arith.subi %c255, %e2 : index
              %y = memref.load %t[%r] : memref<256xf32>
              memref.store %y, %mo[%e2] : memref<256xf32>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[sequential]>}
            scf.reduce
          } {acc.par_dims = #acc<par_dims[thread_x]>}
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>}
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// A buffer private to a worker is shared by the vector lanes of that worker.
// The vector loops are inside a worker loop, where a barrier already follows
// every vector loop, so this check adds none of its own: exactly one barrier
// follows each vector loop. The one gpu.barrier at the end of the worker loop
// body is not from this check.
// CHECK-LABEL: func.func @worker_private_buffer
// CHECK:         scf.for
// CHECK:           memref.store
// CHECK:           nvvm.barrier
// CHECK-NOT:       {{(nvvm|gpu)\.barrier}}
// CHECK:           memref.load
// CHECK:           memref.store
// CHECK:           nvvm.barrier
// CHECK:           gpu.barrier
// CHECK-NOT:       {{(nvvm|gpu)\.barrier}}
// CHECK:         gpu.terminator
func.func @worker_private_buffer(%a: memref<256xf32>, %out: memref<256xf32>,
                                 %nb: index, %nw: index, %nv: index) {
  %bx = acc.par_width %nb par_dim(#acc.par_dim<block_x>)
  %ty = acc.par_width %nw par_dim(#acc.par_dim<thread_y>)
  %tx = acc.par_width %nv par_dim(#acc.par_dim<thread_x>)
  %priv = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<256xf32>>
  acc.compute_region launch(%kbx = %bx, %kty = %ty, %ktx = %tx) ins(%ma = %a, %mo = %out, %mp = %priv) : (memref<256xf32>, memref<256xf32>, !acc.private_type<memref<256xf32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c255 = arith.constant 255 : index
    %c256 = arith.constant 256 : index
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.parallel (%i) = (%g) to (%c32) step (%kbx) {
        scf.parallel (%w) = (%c0) to (%kty) step (%c1) {
          scf.parallel (%j) = (%w) to (%c4) step (%kty) {
            %t = acc.private_local %mp : (!acc.private_type<memref<256xf32>>) -> memref<256xf32>
            scf.for %k = %c0 to %c4 step %c1 {
              scf.parallel (%v) = (%c0) to (%ktx) step (%c1) {
                scf.parallel (%e) = (%v) to (%c256) step (%ktx) {
                  %x = memref.load %ma[%e] : memref<256xf32>
                  memref.store %x, %t[%e] : memref<256xf32>
                  scf.reduce
                } {acc.par_dims = #acc<par_dims[sequential]>}
                scf.reduce
              } {acc.par_dims = #acc<par_dims[thread_x]>}
              scf.parallel (%v2) = (%c0) to (%ktx) step (%c1) {
                scf.parallel (%e2) = (%v2) to (%c256) step (%ktx) {
                  %r = arith.subi %c255, %e2 : index
                  %y = memref.load %t[%r] : memref<256xf32>
                  memref.store %y, %mo[%e2] : memref<256xf32>
                  scf.reduce
                } {acc.par_dims = #acc<par_dims[sequential]>}
                scf.reduce
              } {acc.par_dims = #acc<par_dims[thread_x]>}
            }
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>}
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// The loop writes the buffer through one input of the compute region and the
// code after it reads the same storage through another input, a view of it.
// The region cannot see that, but the values it is passed show that both inputs
// name the same storage, so the read conflicts with the write.
// CHECK-LABEL: func.func @aliased_inputs
// CHECK:         scf.for
// CHECK:           memref.store
// CHECK:           gpu.block_dim x
// CHECK:           gpu.barrier
// CHECK:           memref.load
func.func @aliased_inputs(%a: memref<128xf32>, %b: memref<4xf32>) {
  %c1_pw = arith.constant 1 : index
  %c128_pw = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128_pw par_dim(#acc.par_dim<thread_x>)
  %sub = memref.subview %a[0] [128] [1] : memref<128xf32> to memref<128xf32, strided<[1]>>
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a, %ms = %sub, %mb = %b) : (memref<128xf32>, memref<128xf32, strided<[1]>>, memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %one = arith.constant 1.0 : f32
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.for %it = %c0 to %c4 step %c1 {
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          memref.store %one, %ma[%i] : memref<128xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        %v = memref.load %ms[%c3] : memref<128xf32, strided<[1]>>
        memref.store %v, %mb[%it] : memref<4xf32>
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}
