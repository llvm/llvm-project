// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

// The inlined callsite's call-stack context is concatenated with the cloned
// call's own callsite stack, and the cloned call's MIBs are filtered to the
// ones sharing a prefix with the new stack.

llvm.func @alloc(i64) -> !llvm.ptr

llvm.func @callee(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @alloc(%arg0) {llvm.callsite = array<i64: 222>, llvm.memprof = [#llvm.memprof_mib<call_stack = [222, 111, 444], alloc_type = "cold", context_sizes = <id = 999, size = 4096>>, #llvm.memprof_mib<call_stack = [999], alloc_type = "notcold">]} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @caller
// CHECK: llvm.call @alloc
// CHECK-SAME: llvm.callsite = array<i64: 222, 111>
// CHECK-SAME: llvm.memprof = [#llvm.memprof_mib<call_stack = [222, 111, 444], alloc_type = "cold", context_sizes = <id = 999, size = 4096>>]
llvm.func @caller(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @callee(%arg0) {llvm.callsite = array<i64: 111>} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

// Without callsite metadata on the inlined call, the cloned call keeps
// neither attribute.

llvm.func @alloc(i64) -> !llvm.ptr

llvm.func @callee(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @alloc(%arg0) {llvm.callsite = array<i64: 222>, llvm.memprof = [#llvm.memprof_mib<call_stack = [222, 333], alloc_type = "cold">]} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @caller
// CHECK: llvm.call @alloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK-NOT: llvm.callsite
// CHECK-NOT: llvm.memprof
// CHECK: llvm.return
llvm.func @caller(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @callee(%arg0) : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

// A MIB stack shorter than the new callsite stack still matches when it
// agrees on every element it has, and all MIBs are kept then.

llvm.func @alloc(i64) -> !llvm.ptr

llvm.func @callee(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @alloc(%arg0) {llvm.callsite = array<i64: 222>, llvm.memprof = [#llvm.memprof_mib<call_stack = [222], alloc_type = "cold">]} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @caller
// CHECK: llvm.call @alloc
// CHECK-SAME: llvm.callsite = array<i64: 222, 111>
// CHECK-SAME: llvm.memprof = [#llvm.memprof_mib<call_stack = [222], alloc_type = "cold">]
llvm.func @caller(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @callee(%arg0) {llvm.callsite = array<i64: 111>} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

// A cloned call with a callsite but no memprof only gets its callsite
// stack concatenated.

llvm.func @alloc(i64) -> !llvm.ptr

llvm.func @callee(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @alloc(%arg0) {llvm.callsite = array<i64: 222>} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @caller
// CHECK: llvm.call @alloc
// CHECK-SAME: llvm.callsite = array<i64: 222, 111>
// CHECK-NOT: llvm.memprof
// CHECK: llvm.return
llvm.func @caller(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @callee(%arg0) {llvm.callsite = array<i64: 111>} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

// A cloned call with memprof but no callsite cannot have its context
// extended, so the memprof attribute is dropped.

llvm.func @alloc(i64) -> !llvm.ptr

llvm.func @callee(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @alloc(%arg0) {llvm.memprof = [#llvm.memprof_mib<call_stack = [222, 333], alloc_type = "cold">]} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @caller
// CHECK: llvm.call @alloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK-NOT: llvm.callsite
// CHECK-NOT: llvm.memprof
// CHECK: llvm.return
llvm.func @caller(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @callee(%arg0) {llvm.callsite = array<i64: 111>} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

// When no MIB shares a prefix with the new callsite stack, both attributes
// are removed from the cloned call. The MIB below matches the cloned call's
// own stack but not the concatenated one, so filtering must use the new
// stack.

llvm.func @alloc(i64) -> !llvm.ptr

llvm.func @callee(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @alloc(%arg0) {llvm.callsite = array<i64: 222>, llvm.memprof = [#llvm.memprof_mib<call_stack = [222, 333], alloc_type = "cold">]} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @caller
// CHECK: llvm.call @alloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK-NOT: llvm.callsite
// CHECK-NOT: llvm.memprof
// CHECK: llvm.return
llvm.func @caller(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @callee(%arg0) {llvm.callsite = array<i64: 111>} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

// A malformed memprof list must not crash the inliner: both attributes are
// removed from the cloned call.

llvm.func @alloc(i64) -> !llvm.ptr

llvm.func @callee(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @alloc(%arg0) {llvm.callsite = array<i64: 222>, llvm.memprof = [1 : i32]} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @caller
// CHECK: llvm.call @alloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK-NOT: llvm.callsite
// CHECK-NOT: llvm.memprof
// CHECK: llvm.return
llvm.func @caller(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @callee(%arg0) {llvm.callsite = array<i64: 111>} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

// Inlining twice grows the callsite stack at each level, and the MIB is
// kept when it agrees with the full stack.

llvm.func @alloc(i64) -> !llvm.ptr

llvm.func @inner(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @alloc(%arg0) {llvm.callsite = array<i64: 222>, llvm.memprof = [#llvm.memprof_mib<call_stack = [222, 111, 0, 444], alloc_type = "cold">]} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @middle(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @inner(%arg0) {llvm.callsite = array<i64: 111>} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: llvm.func @caller
// CHECK: llvm.call @alloc
// CHECK-SAME: llvm.callsite = array<i64: 222, 111, 0>
// CHECK-SAME: llvm.memprof = [#llvm.memprof_mib<call_stack = [222, 111, 0, 444], alloc_type = "cold">]
llvm.func @caller(%arg0: i64) -> !llvm.ptr {
  %0 = llvm.call @middle(%arg0) {llvm.callsite = array<i64: 0>} : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
