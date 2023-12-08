// This test checks the introduction of lifetime information for allocas defined
// outside of omp.wsloop and omp.simdloop regions but only used inside of them.

// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @foo(%arg0 : i32) {
  llvm.return
}

llvm.func @bar(%arg0 : i64) {
  llvm.return
}

// CHECK-LABEL: define void @wsloop_i32
llvm.func @wsloop_i32(%size : i64, %lb : i32, %ub : i32, %step : i32) {
  // CHECK-DAG:  %[[LASTITER:.*]] = alloca i32
  // CHECK-DAG:  %[[LB:.*]] = alloca i32
  // CHECK-DAG:  %[[UB:.*]] = alloca i32
  // CHECK-DAG:  %[[STRIDE:.*]] = alloca i32
  // CHECK-DAG:  %[[I:.*]] = alloca i32
  %1 = llvm.alloca %size x i32 : (i64) -> !llvm.ptr

  // CHECK-NOT:  %[[I]]
  // CHECK:      call void @llvm.lifetime.start.p0(i64 4, ptr %[[I]])
  // CHECK-NEXT: br label %[[WSLOOP_BB:.*]]
  // CHECK-NOT:  %[[I]]
  // CHECK:      [[WSLOOP_BB]]:
  // CHECK-NOT:  {{^.*}}:
  // CHECK:      br label %[[CONT_BB:.*]]
  // CHECK:      [[CONT_BB]]:
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 4, ptr %[[I]])
  // CHECK-NOT:  %[[I]]
  omp.wsloop for (%iv) : i32 = (%lb) to (%ub) step (%step) {
    llvm.store %iv, %1 : i32, !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i32
    llvm.call @foo(%2) : (i32) -> ()
    omp.yield
  }

  // CHECK:      ret void
  llvm.return
}

// CHECK-LABEL: define void @wsloop_i64
llvm.func @wsloop_i64(%size : i64, %lb : i64, %ub : i64, %step : i64) {
  // CHECK-DAG:  %[[LASTITER:.*]] = alloca i32
  // CHECK-DAG:  %[[LB:.*]] = alloca i64
  // CHECK-DAG:  %[[UB:.*]] = alloca i64
  // CHECK-DAG:  %[[STRIDE:.*]] = alloca i64
  // CHECK-DAG:  %[[I:.*]] = alloca i64
  %1 = llvm.alloca %size x i64 : (i64) -> !llvm.ptr

  // CHECK-NOT:  %[[I]]
  // CHECK:      call void @llvm.lifetime.start.p0(i64 8, ptr %[[I]])
  // CHECK-NEXT: br label %[[WSLOOP_BB:.*]]
  // CHECK-NOT:  %[[I]]
  // CHECK:      [[WSLOOP_BB]]:
  // CHECK-NOT:  {{^.*}}:
  // CHECK:      br label %[[CONT_BB:.*]]
  // CHECK:      [[CONT_BB]]:
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr %[[I]])
  // CHECK-NOT:  %[[I]]
  omp.wsloop for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    llvm.store %iv, %1 : i64, !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i64
    llvm.call @bar(%2) : (i64) -> ()
    omp.yield
  }

  // CHECK:      ret void
  llvm.return
}

// CHECK-LABEL: define void @simdloop_i32
llvm.func @simdloop_i32(%size : i64, %lb : i32, %ub : i32, %step : i32) {
  // CHECK:      %[[I:.*]] = alloca i32
  %1 = llvm.alloca %size x i32 : (i64) -> !llvm.ptr

  // CHECK-NOT:  %[[I]]
  // CHECK:      call void @llvm.lifetime.start.p0(i64 4, ptr %[[I]])
  // CHECK-NEXT: br label %[[SIMDLOOP_BB:.*]]
  // CHECK-NOT:  %[[I]]
  // CHECK:      [[SIMDLOOP_BB]]:
  // CHECK-NOT:  {{^.*}}:
  // CHECK:      br label %[[CONT_BB:.*]]
  // CHECK:      [[CONT_BB]]:
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 4, ptr %[[I]])
  // CHECK-NOT:  %[[I]]
  omp.simdloop for (%iv) : i32 = (%lb) to (%ub) step (%step) {
    llvm.store %iv, %1 : i32, !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i32
    llvm.call @foo(%2) : (i32) -> ()
    omp.yield
  }

  // CHECK:      ret void
  llvm.return
}

// CHECK-LABEL: define void @simdloop_i64
llvm.func @simdloop_i64(%size : i64, %lb : i64, %ub : i64, %step : i64) {
  // CHECK:      %[[I:.*]] = alloca i64
  %1 = llvm.alloca %size x i64 : (i64) -> !llvm.ptr

  // CHECK-NOT:  %[[I]]
  // CHECK:      call void @llvm.lifetime.start.p0(i64 8, ptr %[[I]])
  // CHECK-NEXT: br label %[[SIMDLOOP_BB:.*]]
  // CHECK-NOT:  %[[I]]
  // CHECK:      [[SIMDLOOP_BB]]:
  // CHECK-NOT:  {{^.*}}:
  // CHECK:      br label %[[CONT_BB:.*]]
  // CHECK:      [[CONT_BB]]:
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr %[[I]])
  // CHECK-NOT:  %[[I]]
  omp.simdloop for (%iv) : i64 = (%lb) to (%ub) step (%step) {
    llvm.store %iv, %1 : i64, !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> i64
    llvm.call @bar(%2) : (i64) -> ()
    omp.yield
  }

  // CHECK:      ret void
  llvm.return
}
