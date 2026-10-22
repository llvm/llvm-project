// RUN: mlir-opt --split-input-file --omp-mark-declare-target %s | FileCheck %s

// declare_target information gets propagated across omp.private.

omp.private {type = firstprivate} @priv : !llvm.struct<(ptr)> init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.call @priv_callee_init() : () -> ()
  omp.yield(%arg1 : !llvm.ptr)
} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.call @priv_callee_copy() : () -> ()
  omp.yield(%arg1 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  llvm.call @priv_callee_dealloc() : () -> ()
  omp.yield
}

// CHECK: llvm.func {{.*}}@priv_callee_nested()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @priv_callee_nested() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@priv_callee_init()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @priv_callee_init() attributes {sym_visibility = "private"} {
  llvm.call @priv_callee_nested() : () -> ()
  llvm.return
}
// CHECK: llvm.func {{.*}}@priv_callee_copy()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @priv_callee_copy() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@priv_callee_dealloc()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @priv_callee_dealloc() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@main()
// CHECK-NOT: omp.declare_target
// CHECK: llvm.return
llvm.func @main() {
  omp.target kernel_type(generic) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    omp.parallel private(@priv %1 -> %arg0 : !llvm.ptr) {
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// declare_target information gets propagated across omp.declare_reduction.

omp.declare_reduction @red : i32
init {
^bb0(%arg0: i32):
  llvm.call @red_callee_init() : () -> ()
  omp.yield (%arg0 : i32)
}
combiner {
^bb1(%arg0: i32, %arg1: i32):
  llvm.call @red_callee_combiner() : () -> ()
  omp.yield (%arg0 : i32)
}
cleanup {
^bb0(%arg0: i32):
  llvm.call @red_callee_cleanup() : () -> ()
  omp.yield
}

// CHECK: llvm.func {{.*}}@red_callee_nested()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to, implicit = true>
llvm.func @red_callee_nested() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@red_callee_init()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to, implicit = true>
llvm.func @red_callee_init() attributes {sym_visibility = "private"} {
  llvm.call @red_callee_nested() : () -> ()
  llvm.return
}
// CHECK: llvm.func {{.*}}@red_callee_combiner()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to, implicit = true>
llvm.func @red_callee_combiner() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@red_callee_cleanup()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to, implicit = true>
llvm.func @red_callee_cleanup() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@main(
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>
llvm.func @main(%arg0 : !llvm.ptr) attributes {
    omp.declare_target = #omp.declaretarget<
        device_type = host, capture_clause = to>} {
  omp.parallel reduction(@red %arg0 -> %arg1 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

// declare_target information gets propagated across the sequence:
// declare_target fn -> fn -> private -> fn -> reduction -> fn -> fn.

omp.private {type = firstprivate} @priv : !llvm.struct<(ptr)> init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.call @priv_callee() : () -> ()
  omp.yield(%arg1 : !llvm.ptr)
} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg1 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  omp.yield
}

omp.declare_reduction @red : i32
init {
^bb0(%arg0: i32):
  llvm.call @red_callee() : () -> ()
  omp.yield (%arg0 : i32)
}
combiner {
^bb1(%arg0: i32, %arg1: i32):
  omp.yield (%arg0 : i32)
}
cleanup {
^bb0(%arg0: i32):
  omp.yield
}

// CHECK: llvm.func {{.*}}@red_callee_nested2()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @red_callee_nested2() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@red_callee_nested()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @red_callee_nested() attributes {sym_visibility = "private"} {
  llvm.call @red_callee_nested2() : () -> ()
  llvm.return
}
// CHECK: llvm.func {{.*}}@red_callee()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @red_callee() attributes {sym_visibility = "private"} {
  llvm.call @red_callee_nested() : () -> ()
  llvm.return
}
// CHECK: llvm.func {{.*}}@priv_callee()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @priv_callee() attributes {sym_visibility = "private"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
  omp.parallel reduction(@red %1 -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}
// CHECK: llvm.func {{.*}}@main_callee()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @main_callee() attributes {sym_visibility = "private"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
  omp.parallel private(@priv %1 -> %arg0 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}
// CHECK: llvm.func {{.*}}@main()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>
llvm.func @main() attributes {
    omp.declare_target = #omp.declaretarget<
        device_type = nohost, capture_clause = to>} {
  llvm.call @main_callee() : () -> ()
  llvm.return
}

// -----

// Non-declare target calling another non-declare target doesn't add any
// attributes.

// CHECK: llvm.func {{.*}}@callee()
// CHECK-NOT: omp.declare_target
// CHECK: llvm.return
llvm.func @callee() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@main()
// CHECK-NOT: omp.declare_target
// CHECK: llvm.return
llvm.func @main() {
  llvm.call @callee() : () -> ()
  llvm.return
}

// -----

// declare_target calling another declare_target doesn't introduce changes.
// If they aren't compatible, this is a user error.

// CHECK: llvm.func {{.*}}@callee()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>
llvm.func @callee() attributes {
    sym_visibility = "private",
    omp.declare_target = #omp.declaretarget<
        device_type = host, capture_clause = to>} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@main()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>
llvm.func @main() attributes {
    omp.declare_target = #omp.declaretarget<
        device_type = nohost, capture_clause = to>} {
  llvm.call @callee() : () -> ()
  llvm.return
}

// -----

// Combining device_type(nohost) and device_type(host) results in
// device_type(any) and it propagates to nested callees.

// CHECK: llvm.func {{.*}}@callee_nested()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>
llvm.func @callee_nested() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@callee()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>
llvm.func @callee() attributes {sym_visibility = "private"} {
  llvm.call @callee_nested() : () -> ()
  llvm.return
}
// CHECK: llvm.func {{.*}}@fn_host()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>
llvm.func @fn_host() attributes {
    omp.declare_target = #omp.declaretarget<
        device_type = host, capture_clause = to>} {
  llvm.call @callee() : () -> ()
  llvm.return
}
// CHECK: llvm.func {{.*}}@fn_nohost()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>
llvm.func @fn_nohost() attributes {
    omp.declare_target = #omp.declaretarget<
        device_type = nohost, capture_clause = to>} {
  llvm.call @callee() : () -> ()
  llvm.return
}

// -----

// Always use implicit device_type(any) for external and public functions.

// CHECK: llvm.func {{.*}}@external_host()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>
llvm.func @external_host()
// CHECK: llvm.func {{.*}}@external_nohost()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>
llvm.func @external_nohost()
// CHECK: llvm.func {{.*}}@external_both()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>
llvm.func @external_both()
// CHECK: llvm.func {{.*}}@public_host()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>
llvm.func @public_host() {
  llvm.return
}
// CHECK: llvm.func {{.*}}@public_nohost()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>
llvm.func @public_nohost() {
  llvm.return
}
// CHECK: llvm.func {{.*}}@public_both()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>
llvm.func @public_both() {
  llvm.return
}
// CHECK: llvm.func {{.*}}@fn_host()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>
llvm.func @fn_host() attributes {
    omp.declare_target = #omp.declaretarget<
        device_type = host, capture_clause = to>} {
  llvm.call @external_host() : () -> ()
  llvm.call @external_both() : () -> ()
  llvm.call @public_host() : () -> ()
  llvm.call @public_both() : () -> ()
  llvm.return
}
// CHECK: llvm.func {{.*}}@fn_nohost()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>
llvm.func @fn_nohost() attributes {
    omp.declare_target = #omp.declaretarget<
        device_type = nohost, capture_clause = to>} {
  llvm.call @external_nohost() : () -> ()
  llvm.call @external_both() : () -> ()
  llvm.call @public_nohost() : () -> ()
  llvm.call @public_both() : () -> ()
  llvm.return
}

// -----

// omp.target private propagates device_type(nohost), unlike in_reduction.

omp.private {type = firstprivate} @priv : !llvm.struct<(ptr)> init {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  llvm.call @priv_callee() : () -> ()
  omp.yield(%arg1 : !llvm.ptr)
} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg1 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  omp.yield
}

omp.declare_reduction @red : i32
init {
^bb0(%arg0: i32):
  llvm.call @red_callee() : () -> ()
  omp.yield (%arg0 : i32)
}
combiner {
^bb1(%arg0: i32, %arg1: i32):
  omp.yield (%arg0 : i32)
}
cleanup {
^bb0(%arg0: i32):
  omp.yield
}

// CHECK: llvm.func {{.*}}@red_callee()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to, implicit = true>
llvm.func @red_callee() attributes {sym_visibility = "private"} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@priv_callee()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>
llvm.func @priv_callee() attributes {sym_visibility = "private"} {
  llvm.return
}

llvm.func @main(%arg0 : !llvm.ptr) attributes {
    omp.declare_target = #omp.declaretarget<
        device_type = host, capture_clause = to>} {
  %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
  omp.target kernel_type(generic) in_reduction(@red %arg0 : !llvm.ptr)
             map_entries(%0 -> %arg1 : !llvm.ptr)
             private(@priv %arg0 -> %arg2 : !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}

// -----

// Implicit declare_target information should be updatable by another run of the
// pass.

// CHECK: llvm.func {{.*}}@callee()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>
llvm.func @callee() attributes {
    sym_visibility = "private",
    omp.declare_target = #omp.declaretarget<
        device_type = host, capture_clause = to, implicit = true>} {
  llvm.return
}
// CHECK: llvm.func {{.*}}@main()
// CHECK-SAME: omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>
llvm.func @main() attributes {
    omp.declare_target = #omp.declaretarget<
        device_type = nohost, capture_clause = to>} {
  llvm.call @callee() : () -> ()
  llvm.return
}
