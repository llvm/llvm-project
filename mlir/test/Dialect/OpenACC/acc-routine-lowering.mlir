// RUN: mlir-opt %s -acc-routine-lowering -split-input-file | FileCheck %s

// Test seq routine: body is wrapped in acc.compute_region with one
// acc.par_width (sequential).
acc.routine @routine_seq func(@host_foo) seq
// CHECK: acc.routine @routine_seq func(@
// CHECK: acc.specialized_routine = #acc.specialized_routine<@routine_seq, <seq>, "host_foo">
// CHECK-NOT: acc.kernel_environment
// CHECK: acc.par_width {par_dim = #acc.par_dim<sequential>}
// CHECK: acc.compute_region
// CHECK: origin = "acc.routine"
func.func @host_foo(%buf: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  memref.store %c0_i32, %buf[%c0] : memref<8xi32>
  return
}

// -----

// Test vector routine: par_width has thread_x with default mapping of vector dimension.
acc.routine @routine_vec func(@host_bar) vector
// CHECK: acc.routine @routine_vec func(@
// CHECK: acc.specialized_routine = #acc.specialized_routine<@routine_vec, <vector>, "host_bar">
// CHECK-NOT: acc.kernel_environment
// CHECK: acc.par_width {par_dim = #acc.par_dim<thread_x>}
// CHECK: acc.compute_region
// CHECK: origin = "acc.routine"
func.func @host_bar(%buf: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  memref.store %c0_i32, %buf[%c0] : memref<4xi32>
  return
}

// -----

// Test worker routine: par_width has thread_y with default mapping of worker dimension.
acc.routine @routine_worker func(@host_worker) worker
// CHECK: acc.routine @routine_worker func(@
// CHECK: acc.specialized_routine = #acc.specialized_routine<@routine_worker, <worker>, "host_worker">
// CHECK: acc.par_width {par_dim = #acc.par_dim<thread_y>}
// CHECK: acc.compute_region
// CHECK: origin = "acc.routine"
func.func @host_worker(%x: i32) {
  return
}

// -----

// Test gang routine: par_width has block_x (gang dim 1) with default mapping of gang dimension.
acc.routine @routine_gang func(@host_gang) gang
// CHECK: acc.routine @routine_gang func(@
// CHECK: acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "host_gang">
// CHECK: acc.par_width {par_dim = #acc.par_dim<block_x>}
// CHECK: acc.compute_region
// CHECK: origin = "acc.routine"
func.func @host_gang() {
  return
}

// -----

// Test routine with single return value: device func returns compute_region result.
acc.routine @routine_ret func(@host_ret) seq
// CHECK: acc.routine @routine_ret func(@
// CHECK: acc.specialized_routine = #acc.specialized_routine<@routine_ret, <seq>, "host_ret">
// CHECK: %[[CR:[0-9]+]] = acc.compute_region
// CHECK: acc.yield %{{.*}} : i32
// CHECK: return %[[CR]] : i32
func.func @host_ret(%cond: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = arith.select %cond, %c0, %c1 : i32
  return %r : i32
}

// -----

// Test routine with unstructured control flow.
acc.routine @routine_cf func(@host_cf) seq
// CHECK: acc.routine @routine_cf func(@
// CHECK: acc.specialized_routine = #acc.specialized_routine<@routine_cf, <seq>, "host_cf">
// CHECK: acc.par_width {par_dim = #acc.par_dim<sequential>}
// CHECK: %[[CR:[0-9]+]] = acc.compute_region
// CHECK: %[[EXE:[0-9]+]] = scf.execute_region
// CHECK: scf.yield %{{.*}} : i32
// CHECK: acc.yield %[[EXE]] : i32
// CHECK: } {origin = "acc.routine"}
// CHECK: return %[[CR]] : i32
func.func @host_cf(%cond: i1) -> i32 {
  cf.cond_br %cond, ^then, ^else
^then:
  %c0 = arith.constant 0 : i32
  return %c0 : i32
^else:
  %c1 = arith.constant 1 : i32
  return %c1 : i32
}

// -----

// Test routine with bind(name): pass skips it, routine and func remain unchanged.
acc.routine @routine_bind func(@host_bind) seq bind("myname")
// CHECK: acc.routine @routine_bind func(@host_bind)
// CHECK-NOT: acc.specialized_routine
// CHECK: func.func @host_bind()
func.func @host_bind() {
  return
}

// -----

// Test multiple routines in one module: each gets its own device copy.
acc.routine @r_a func(@f_a) seq
acc.routine @r_b func(@f_a) vector
// CHECK: acc.routine @r_a func(@
// CHECK: acc.specialized_routine = #acc.specialized_routine<@r_a, <seq>, "f_a">
// CHECK: acc.par_width {par_dim = #acc.par_dim<sequential>}
// CHECK: acc.routine @r_b func(@
// CHECK: acc.specialized_routine = #acc.specialized_routine<@r_b, <vector>, "f_a">
// CHECK: acc.par_width {par_dim = #acc.par_dim<thread_x>}
func.func @f_a() {
  return
}
