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

llvm.func @dynamic_affinity(%x: !llvm.ptr, %lb: i64, %ub: i64,
                            %step: i64) {
  %len = llvm.mlir.constant(4 : i64) : i64
  %it = omp.iterator(%i: i64) = (%lb to %ub step %step) {
    %entry = omp.affinity_entry %x, %len
        : (!llvm.ptr, i64) -> !omp.affinity_entry_ty<!llvm.ptr, i64>
    omp.yield(%entry : !omp.affinity_entry_ty<!llvm.ptr, i64>)
  } -> !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>
  omp.task affinity(
      %it : !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @dynamic_affinity(
// CHECK: %[[START:.*]] = sext i64 %{{.*}} to i65
// CHECK: %[[STOP:.*]] = sext i64 %{{.*}} to i65
// CHECK: %[[STEP:.*]] = sext i64 %{{.*}} to i65
// CHECK: %[[NEG:.*]] = icmp slt i65 %[[STEP]], 0
// CHECK: %[[INCR:.*]] = select i1 %[[NEG]], i65 %{{.*}}, i65 %[[STEP]]
// CHECK: %[[LB:.*]] = select i1 %[[NEG]], i65 %[[STOP]], i65 %[[START]]
// CHECK: %[[UB:.*]] = select i1 %[[NEG]], i65 %[[START]], i65 %[[STOP]]
// CHECK: %[[SPAN:.*]] = sub nsw i65 %[[UB]], %[[LB]]
// CHECK: %[[EMPTY:.*]] = icmp slt i65 %[[UB]], %[[LB]]
// CHECK: %[[DIV:.*]] = udiv i65 %[[SPAN]], %[[INCR]]
// CHECK: %[[COUNT:.*]] = add i65 %[[DIV]], 1
// CHECK: %[[CLAMPED:.*]] = select i1 %[[EMPTY]], i65 0, i65 %[[COUNT]]
// CHECK: %[[TRIPS:.*]] = trunc i65 %[[CLAMPED]] to i64
// CHECK: %[[TOTAL:.*]] = mul i64 1, %[[TRIPS]]
// CHECK: alloca {{.*}}, i64 %[[TOTAL]]
// CHECK: icmp ult i64 %{{.*}}, %[[TOTAL]]
// CHECK: trunc i64 %[[TOTAL]] to i32
