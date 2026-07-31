// RUN: mlir-opt %s -buffer-loop-merging -split-input-file | FileCheck %s

// A memref iter_arg whose init and yielded values are distinct allocas that are
// used interchangeably across iterations is merged onto the init buffer, making
// the iter_arg loop-invariant.

// CHECK-LABEL: func.func @merge_converging_buffers
//       CHECK:   memref.alloca() : memref<f32>
//       CHECK:   %[[INIT:.*]] = memref.alloca() : memref<f32>
//       CHECK:   memref.store %{{.*}}, %[[INIT]][]
//       CHECK:   scf.for {{.*}} iter_args(%{{.*}} = %[[INIT]])
//       CHECK:     memref.store %{{.*}}, %[[INIT]][]
//       CHECK:     scf.yield %[[INIT]]
func.func @merge_converging_buffers(%init: f32, %lb: index, %ub: index, %st: index) -> f32 {
  %alloc_yield = memref.alloca() : memref<f32>
  %alloc_init = memref.alloca() : memref<f32>
  memref.store %init, %alloc_init[] : memref<f32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%lv = %alloc_init) -> (memref<f32>) {
    %v = memref.load %lv[] : memref<f32>
    %n = arith.addf %v, %v : f32
    memref.store %n, %alloc_yield[] : memref<f32>
    scf.yield %alloc_yield : memref<f32>
  }
  %o = memref.load %r[] : memref<f32>
  return %o : f32
}

// -----

// Same pattern with whole-buffer vector transfers instead of load/store.

// CHECK-LABEL: func.func @merge_converging_buffers_vector
//       CHECK:   memref.alloca() : memref<128xf32>
//       CHECK:   %[[INIT:.*]] = memref.alloca() : memref<128xf32>
//       CHECK:   vector.transfer_write %{{.*}}, %[[INIT]]
//       CHECK:   scf.for {{.*}} iter_args(%{{.*}} = %[[INIT]])
//       CHECK:     vector.transfer_write %{{.*}}, %[[INIT]]
//       CHECK:     scf.yield %[[INIT]]
func.func @merge_converging_buffers_vector(%pad: f32, %init: vector<128xf32>,
                                           %lb: index, %ub: index, %st: index) -> vector<128xf32> {
  %c0 = arith.constant 0 : index
  %alloc_yield = memref.alloca() : memref<128xf32>
  %alloc_init = memref.alloca() : memref<128xf32>
  vector.transfer_write %init, %alloc_init[%c0] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%lv = %alloc_init) -> (memref<128xf32>) {
    %v = vector.transfer_read %lv[%c0], %pad {in_bounds = [true]} : memref<128xf32>, vector<128xf32>
    %n = arith.addf %v, %v : vector<128xf32>
    vector.transfer_write %n, %alloc_yield[%c0] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
    scf.yield %alloc_yield : memref<128xf32>
  }
  %o = vector.transfer_read %r[%c0], %pad {in_bounds = [true]} : memref<128xf32>, vector<128xf32>
  return %o : vector<128xf32>
}

// -----

// A genuine ping-pong swap rotates both buffers every iteration, so which buffer
// a given iteration reads is not statically known. Must not merge.

// CHECK-LABEL: func.func @no_merge_ping_pong
//       CHECK:   scf.yield %{{.*}}, %{{.*}} : memref<f32>, memref<f32>
func.func @no_merge_ping_pong(%init: f32, %lb: index, %ub: index, %st: index) -> f32 {
  %a = memref.alloca() : memref<f32>
  %b = memref.alloca() : memref<f32>
  memref.store %init, %a[] : memref<f32>
  %r:2 = scf.for %i = %lb to %ub step %st iter_args(%src = %a, %dst = %b) -> (memref<f32>, memref<f32>) {
    %v = memref.load %src[] : memref<f32>
    %n = arith.addf %v, %v : f32
    memref.store %n, %dst[] : memref<f32>
    scf.yield %dst, %src : memref<f32>, memref<f32>
  }
  %o = memref.load %r#0[] : memref<f32>
  return %o : f32
}

// -----

// The yielded buffer is also read inside the body, so merging would make that
// read observe the current iteration's store. Must not merge.

// CHECK-LABEL: func.func @no_merge_yielded_is_read
//       CHECK:   %[[YIELD:.*]] = memref.alloca() : memref<f32>
//       CHECK:   memref.load %[[YIELD]][]
//       CHECK:   scf.yield %[[YIELD]]
func.func @no_merge_yielded_is_read(%init: f32, %lb: index, %ub: index, %st: index) -> f32 {
  %alloc_yield = memref.alloca() : memref<f32>
  %alloc_init = memref.alloca() : memref<f32>
  memref.store %init, %alloc_init[] : memref<f32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%lv = %alloc_init) -> (memref<f32>) {
    %v = memref.load %lv[] : memref<f32>
    %prev = memref.load %alloc_yield[] : memref<f32>
    %n = arith.addf %v, %prev : f32
    memref.store %n, %alloc_yield[] : memref<f32>
    scf.yield %alloc_yield : memref<f32>
  }
  %o = memref.load %r[] : memref<f32>
  return %o : f32
}

// -----

// Heap allocations may be freed or aliased elsewhere; only allocas are merged.

// CHECK-LABEL: func.func @no_merge_heap_alloc
//       CHECK:   %[[YIELD:.*]] = memref.alloc() : memref<f32>
//       CHECK:   scf.yield %[[YIELD]]
func.func @no_merge_heap_alloc(%init: f32, %lb: index, %ub: index, %st: index) -> f32 {
  %alloc_yield = memref.alloc() : memref<f32>
  %alloc_init = memref.alloc() : memref<f32>
  memref.store %init, %alloc_init[] : memref<f32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%lv = %alloc_init) -> (memref<f32>) {
    %v = memref.load %lv[] : memref<f32>
    %n = arith.addf %v, %v : f32
    memref.store %n, %alloc_yield[] : memref<f32>
    scf.yield %alloc_yield : memref<f32>
  }
  %o = memref.load %r[] : memref<f32>
  return %o : f32
}

// -----

// The yielded buffer escapes the function, so its identity is observable.

// CHECK-LABEL: func.func @no_merge_escaping_buffer
//       CHECK:   %[[YIELD:.*]] = memref.alloca() : memref<f32>
//       CHECK:   return %[[YIELD]]
func.func @no_merge_escaping_buffer(%init: f32, %lb: index, %ub: index, %st: index) -> memref<f32> {
  %alloc_yield = memref.alloca() : memref<f32>
  %alloc_init = memref.alloca() : memref<f32>
  memref.store %init, %alloc_init[] : memref<f32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%lv = %alloc_init) -> (memref<f32>) {
    %v = memref.load %lv[] : memref<f32>
    %n = arith.addf %v, %v : f32
    memref.store %n, %alloc_yield[] : memref<f32>
    scf.yield %alloc_yield : memref<f32>
  }
  return %alloc_yield : memref<f32>
}

// -----

// The init buffer is read *after* the loop. Merging redirects the body's stores
// into the init buffer, so that post-loop read would observe the last
// iteration's value instead of the original init contents. Must not merge.

// CHECK-LABEL: func.func @no_merge_init_read_after_loop
//       CHECK:   %[[YIELD:.*]] = memref.alloca() : memref<f32>
//       CHECK:   %[[INIT:.*]] = memref.alloca() : memref<f32>
//       CHECK:   scf.yield %[[YIELD]]
//       CHECK:   memref.load %[[INIT]][]
func.func @no_merge_init_read_after_loop(%init: f32, %lb: index, %ub: index, %st: index) -> (f32, f32) {
  %alloc_yield = memref.alloca() : memref<f32>
  %alloc_init = memref.alloca() : memref<f32>
  memref.store %init, %alloc_init[] : memref<f32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%lv = %alloc_init) -> (memref<f32>) {
    %v = memref.load %lv[] : memref<f32>
    %n = arith.addf %v, %v : f32
    memref.store %n, %alloc_yield[] : memref<f32>
    scf.yield %alloc_yield : memref<f32>
  }
  %x = memref.load %alloc_init[] : memref<f32>
  %o = memref.load %r[] : memref<f32>
  return %x, %o : f32, f32
}

// -----

// The yielded buffer is written *before* the loop. Merging redirects that store
// into the init buffer too, changing which initial value the loop reads. Must
// not merge.

// CHECK-LABEL: func.func @no_merge_yield_written_before_loop
//       CHECK:   %[[YIELD:.*]] = memref.alloca() : memref<f32>
//       CHECK:   %[[INIT:.*]] = memref.alloca() : memref<f32>
//       CHECK:   memref.store %{{.*}}, %[[YIELD]][]
//       CHECK:   scf.for {{.*}} iter_args(%{{.*}} = %[[INIT]])
func.func @no_merge_yield_written_before_loop(%i0: f32, %i1: f32, %lb: index, %ub: index, %st: index) -> f32 {
  %alloc_yield = memref.alloca() : memref<f32>
  %alloc_init = memref.alloca() : memref<f32>
  memref.store %i0, %alloc_init[] : memref<f32>
  memref.store %i1, %alloc_yield[] : memref<f32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%lv = %alloc_init) -> (memref<f32>) {
    %v = memref.load %lv[] : memref<f32>
    %n = arith.addf %v, %v : f32
    memref.store %n, %alloc_yield[] : memref<f32>
    scf.yield %alloc_yield : memref<f32>
  }
  %o = memref.load %r[] : memref<f32>
  return %o : f32
}

// -----

// The yielded buffer may be read after the loop: after merging it aliases the
// init buffer, which holds the same bytes, so the read is preserved. Merges.

// CHECK-LABEL: func.func @merge_yield_read_after_loop
//       CHECK:   memref.alloca() : memref<f32>
//       CHECK:   %[[INIT:.*]] = memref.alloca() : memref<f32>
//       CHECK:   scf.for {{.*}} iter_args(%{{.*}} = %[[INIT]])
//       CHECK:     scf.yield %[[INIT]]
//       CHECK:   memref.load %[[INIT]][]
func.func @merge_yield_read_after_loop(%init: f32, %lb: index, %ub: index, %st: index) -> f32 {
  %alloc_yield = memref.alloca() : memref<f32>
  %alloc_init = memref.alloca() : memref<f32>
  memref.store %init, %alloc_init[] : memref<f32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%lv = %alloc_init) -> (memref<f32>) {
    %v = memref.load %lv[] : memref<f32>
    %n = arith.addf %v, %v : f32
    memref.store %n, %alloc_yield[] : memref<f32>
    scf.yield %alloc_yield : memref<f32>
  }
  %o = memref.load %alloc_yield[] : memref<f32>
  return %o : f32
}
