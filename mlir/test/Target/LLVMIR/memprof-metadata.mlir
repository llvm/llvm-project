// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @alloc(i64) -> !llvm.ptr

// CHECK-LABEL: @simple_memprof
llvm.func @simple_memprof() -> !llvm.ptr {
  %0 = llvm.mlir.constant(16 : i64) : i64
  // CHECK: call ptr @alloc({{.*}}), !memprof ![[MEMPROF:[0-9]+]], !callsite ![[STACK:[0-9]+]]
  %1 = llvm.call @alloc(%0) {
    llvm.callsite = array<i64: 123456789>,
    llvm.memprof = [#llvm.memprof_mib<call_stack = [123456789], alloc_type = "cold">]
  } : (i64) -> !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// CHECK-DAG: ![[MEMPROF]] = !{![[MIB:[0-9]+]]}
// CHECK-DAG: ![[MIB]] = !{![[STACK]], !"cold"}
// CHECK-DAG: ![[STACK]] = !{i64 123456789}

// -----

llvm.func @alloc(i64) -> !llvm.ptr

// One MIB carries two (context id, total size) pairs; the second id has the
// top bit set, locking in the bit-pattern path through the translation.

// CHECK-LABEL: @rich_memprof
llvm.func @rich_memprof(%arg0: i64) -> !llvm.ptr {
  // CHECK: call ptr @alloc({{.*}}), !memprof ![[MEMPROF:[0-9]+]], !callsite ![[STACK2:[0-9]+]]
  %0 = llvm.call @alloc(%arg0) {
    llvm.callsite = array<i64: 333>,
    llvm.memprof = [
      #llvm.memprof_mib<call_stack = [111, -6528110295079665978], alloc_type = "cold",
                        context_sizes = <id = 999, size = 4096>,
                                        <id = 18446744073709551614, size = 8>>,
      #llvm.memprof_mib<call_stack = [333], alloc_type = "notcold">]
  } : (i64) -> !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK-DAG: ![[MEMPROF]] = !{![[MIB1:[0-9]+]], ![[MIB2:[0-9]+]]}
// CHECK-DAG: ![[MIB1]] = !{![[STACK1:[0-9]+]], !"cold", ![[SIZES1:[0-9]+]], ![[SIZES2:[0-9]+]]}
// CHECK-DAG: ![[STACK1]] = !{i64 111, i64 -6528110295079665978}
// CHECK-DAG: ![[SIZES1]] = !{i64 999, i64 4096}
// CHECK-DAG: ![[SIZES2]] = !{i64 -2, i64 8}
// CHECK-DAG: ![[MIB2]] = !{![[STACK2]], !"notcold"}
// CHECK-DAG: ![[STACK2]] = !{i64 333}

// -----

llvm.func @alloc(i64) -> !llvm.ptr

// CHECK-LABEL: @memprof_on_invoke
llvm.func @__gxx_personality_v0(...) -> i32

llvm.func @memprof_on_invoke() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(16 : i64) : i64
  // CHECK: invoke ptr @alloc(i64 16)
  // CHECK-NEXT: to label %{{.*}} unwind label %{{.*}}, !memprof ![[MEMPROF:[0-9]+]], !callsite ![[STACK:[0-9]+]]
  %1 = llvm.invoke @alloc(%0) to ^bb1 unwind ^bb2 {
    llvm.callsite = array<i64: 7>,
    llvm.memprof = [#llvm.memprof_mib<call_stack = [7], alloc_type = "hot">]
  } : (i64) -> !llvm.ptr
^bb1:
  llvm.return %1 : !llvm.ptr
^bb2:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  %null = llvm.mlir.zero : !llvm.ptr
  llvm.return %null : !llvm.ptr
}

// CHECK-DAG: ![[MEMPROF]] = !{![[MIB:[0-9]+]]}
// CHECK-DAG: ![[MIB]] = !{![[STACK]], !"hot"}
// CHECK-DAG: ![[STACK]] = !{i64 7}

// -----

llvm.func @alloc(i64) -> !llvm.ptr

// A call that only carries `!callsite`: the common form for intermediate
// frames, which have no allocation of their own to record.

// CHECK-LABEL: @callsite_only
llvm.func @callsite_only() -> !llvm.ptr {
  %0 = llvm.mlir.constant(16 : i64) : i64
  // CHECK: call ptr @alloc({{.*}}), !callsite ![[STACK:[0-9]+]]
  // CHECK-NOT: !memprof
  %1 = llvm.call @alloc(%0) {
    llvm.callsite = array<i64: 111, 222>
  } : (i64) -> !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// CHECK-DAG: ![[STACK]] = !{i64 111, i64 222}
