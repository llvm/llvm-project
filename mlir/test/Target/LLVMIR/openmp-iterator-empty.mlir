// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// Inclusive ranges must be checked before dividing: (2 - 3) / 2 + 1
// truncates to one even though the range is empty.
llvm.func @empty_positive(%x: !llvm.ptr) {
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %it = omp.iterator(%i: i64) = (%c3 to %c2 step %c2) {
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.taskwait depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>)
  llvm.return
}

// CHECK-LABEL: define void @empty_positive(
// CHECK: call ptr @malloc(i64 0)
// CHECK: icmp ult i64 %{{.*}}, 0
// CHECK: call void @__kmpc_omp_taskwait_deps_51(
// CHECK-SAME: i32 0, ptr %{{.*}}, i32 0, ptr null, i32 0)

llvm.func @empty_negative(%x: !llvm.ptr) {
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %step = llvm.mlir.constant(-2 : i64) : i64
  %it = omp.iterator(%i: i64) = (%c2 to %c3 step %step) {
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.taskwait depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>)
  llvm.return
}

// CHECK-LABEL: define void @empty_negative(
// CHECK: call ptr @malloc(i64 0)
// CHECK: icmp ult i64 %{{.*}}, 0
// CHECK: call void @__kmpc_omp_taskwait_deps_51(
// CHECK-SAME: i32 0, ptr %{{.*}}, i32 0, ptr null, i32 0)

// Two negative counts must not multiply into a nonempty iteration space.
llvm.func @empty_product(%x: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %it = omp.iterator(%i: i64, %j: i64) =
      (%c3 to %c1 step %c1, %c3 to %c1 step %c1) {
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.taskwait depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>)
  llvm.return
}

// CHECK-LABEL: define void @empty_product(
// CHECK: call ptr @malloc(i64 0)
// CHECK: icmp ult i64 %{{.*}}, 0
// CHECK: call void @__kmpc_omp_taskwait_deps_51(
// CHECK-SAME: i32 0, ptr %{{.*}}, i32 0, ptr null, i32 0)
