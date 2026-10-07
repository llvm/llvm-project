// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// Bounds and induction values keep their declared width; only the trip count
// is converted to i64.
llvm.func @dynamic(%x: !llvm.ptr, %lb: i128, %ub: i128, %st: i128) {
  %it = omp.iterator(%i: i128) = (%lb to %ub step %st) {
    llvm.store %i, %x : i128, !llvm.ptr
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.taskwait depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>)
  llvm.return
}
// CHECK-LABEL: define void @dynamic(
// CHECK-SAME: i128 %[[LB:[0-9]+]], i128 %{{[0-9]+}}, i128 %[[ST:[0-9]+]])
// CHECK: %[[IDX:.*]] = zext i64 %{{.*}} to i128
// CHECK: %[[OFFSET:.*]] = mul i128 %[[IDX]], %[[ST]]
// CHECK: %[[IV:.*]] = add i128 %[[LB]], %[[OFFSET]]
// CHECK: store i128 %[[IV]], ptr

// The unsigned linear index must also convert to narrower region arguments.
llvm.func @narrow(%x: !llvm.ptr, %lb: i32, %ub: i32, %st: i32) {
  %it = omp.iterator(%i: i32) = (%lb to %ub step %st) {
    llvm.store %i, %x : i32, !llvm.ptr
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.taskwait depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>)
  llvm.return
}
// CHECK-LABEL: define void @narrow(
// CHECK-SAME: i32 %[[LB:[0-9]+]], i32 %{{[0-9]+}}, i32 %[[ST:[0-9]+]])
// CHECK: %[[IDX:.*]] = trunc i64 %{{.*}} to i32
// CHECK: %[[OFFSET:.*]] = mul i32 %[[IDX]], %[[ST]]
// CHECK: %[[IV:.*]] = add i32 %[[LB]], %[[OFFSET]]
// CHECK: store i32 %[[IV]], ptr

llvm.func @nonempty_positive(%x: !llvm.ptr) {
  %lb = llvm.mlir.constant(18446744073709551617 : i128) : i128
  %ub = llvm.mlir.constant(18446744073709551619 : i128) : i128
  %st = llvm.mlir.constant(1 : i128) : i128
  %it = omp.iterator(%i: i128) = (%lb to %ub step %st) {
    llvm.store %i, %x : i128, !llvm.ptr
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.taskwait depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>)
  llvm.return
}
// CHECK-LABEL: define void @nonempty_positive(
// CHECK: icmp ult i64 %{{.*}}, 3
// CHECK: %[[IDX:.*]] = zext i64 %{{.*}} to i128
// CHECK: %[[OFFSET:.*]] = mul i128 %[[IDX]], 1
// CHECK: %[[IV:.*]] = add i128 18446744073709551617, %[[OFFSET]]
// CHECK: store i128 %[[IV]], ptr
// CHECK: call void @__kmpc_omp_taskwait_deps_51(
// CHECK-SAME: i32 3, ptr %{{.*}}, i32 0, ptr null, i32 0)

llvm.func @nonempty_negative(%x: !llvm.ptr) {
  %lb = llvm.mlir.constant(18446744073709551619 : i128) : i128
  %ub = llvm.mlir.constant(18446744073709551617 : i128) : i128
  %st = llvm.mlir.constant(-1 : i128) : i128
  %it = omp.iterator(%i: i128) = (%lb to %ub step %st) {
    llvm.store %i, %x : i128, !llvm.ptr
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.taskwait depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>)
  llvm.return
}
// CHECK-LABEL: define void @nonempty_negative(
// CHECK: icmp ult i64 %{{.*}}, 3
// CHECK: %[[IDX:.*]] = zext i64 %{{.*}} to i128
// CHECK: %[[OFFSET:.*]] = mul i128 %[[IDX]], -1
// CHECK: %[[IV:.*]] = add i128 18446744073709551619, %[[OFFSET]]
// CHECK: store i128 %[[IV]], ptr
// CHECK: call void @__kmpc_omp_taskwait_deps_51(
// CHECK-SAME: i32 3, ptr %{{.*}}, i32 0, ptr null, i32 0)
