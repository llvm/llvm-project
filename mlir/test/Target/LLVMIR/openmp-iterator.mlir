// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

llvm.func @task_affinity_iterator_1d(%arr: !llvm.ptr {llvm.nocapture}) {
  %c1  = llvm.mlir.constant(1 : i64) : i64
  %c4  = llvm.mlir.constant(4 : i64) : i64
  %c6  = llvm.mlir.constant(6 : i64) : i64
  %len = llvm.mlir.constant(4 : i64) : i64

  omp.parallel {
    omp.single {
      %it = omp.iterator(%i: i64, %j: i64) =
          (%c1 to %c4 step %c1, %c1 to %c6 step %c1) {
        %entry = omp.affinity_entry %arr, %len
            : (!llvm.ptr, i64) -> !omp.affinity_entry_ty<!llvm.ptr, i64>
        omp.yield(%entry : !omp.affinity_entry_ty<!llvm.ptr, i64>)
      } -> !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>

      omp.task affinity(%it : !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>) {
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @task_affinity_iterator_1d

// Preheader -> Header
// CHECK: omp_iterator.preheader:
// CHECK: br label %omp_iterator.header
//
// Header has the IV phi and branches to cond
// CHECK: omp_iterator.header:
// CHECK: [[IV:%.*]] = phi i64 [ 0, %omp_iterator.preheader ], [ [[NEXT:%.*]], %omp_iterator.inc ]
// CHECK: br label %omp_iterator.cond
//
// Cond: IV < 24 and branches to body or exit
// CHECK: omp_iterator.cond:
// CHECK: [[CMP:%.*]] = icmp ult i64 [[IV]], 24
// CHECK: br i1 [[CMP]], label %omp_iterator.body, label %omp_iterator.exit
//
// Exit -> After -> continuation
// CHECK: omp_iterator.exit:
// CHECK: br label %omp_iterator.after
// CHECK: omp_iterator.after:
// CHECK: br label %omp.it.cont
//
// Body: store into affinity_list[IV] then branch to inc
// CHECK: omp_iterator.body:
// CHECK: [[ENTRY:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr %loadgep_omp.affinity_list, i64 [[IV]]
// CHECK: [[ADDRI64:%.*]] = ptrtoint ptr %loadgep_ to i64
// CHECK: [[ADDRGEP:%.*]] = getelementptr inbounds nuw { i64, i64, i32 }, ptr [[ENTRY]], i32 0, i32 0
// CHECK: store i64 [[ADDRI64]], ptr [[ADDRGEP]]
// CHECK: [[LENGEP:%.*]] = getelementptr inbounds nuw { i64, i64, i32 }, ptr [[ENTRY]], i32 0, i32 1
// CHECK: store i64 4, ptr [[LENGEP]]
// CHECK: [[FLAGGEP:%.*]] = getelementptr inbounds nuw { i64, i64, i32 }, ptr [[ENTRY]], i32 0, i32 2
// CHECK: store i32 0, ptr [[FLAGGEP]]
// CHECK: br label %omp_iterator.inc
//
// CHECK: omp_iterator.inc:
// CHECK: [[NEXT]] = add nuw i64 [[IV]], 1
// CHECK: br label %omp_iterator.header

llvm.func @task_affinity_iterator_3d(%arr: !llvm.ptr {llvm.nocapture}) {
  %c1  = llvm.mlir.constant(1 : i64) : i64
  %c2  = llvm.mlir.constant(2 : i64) : i64
  %c4  = llvm.mlir.constant(4 : i64) : i64
  %c6  = llvm.mlir.constant(6 : i64) : i64
  %len = llvm.mlir.constant(4 : i64) : i64

  omp.parallel {
    omp.single {
      // 3-D iterator: i=1..4, j=1..6, k=1..2 => total trips = 48
      %it = omp.iterator(%i: i64, %j: i64, %k: i64) =
          (%c1 to %c4 step %c1, %c1 to %c6 step %c1, %c1 to %c2 step %c1) {
        %entry = omp.affinity_entry %arr, %len
            : (!llvm.ptr, i64) -> !omp.affinity_entry_ty<!llvm.ptr, i64>
        omp.yield(%entry : !omp.affinity_entry_ty<!llvm.ptr, i64>)
      } -> !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>

      omp.task affinity(%it : !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>) {
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @task_affinity_iterator_3d

// Preheader -> Header
// CHECK: omp_iterator.preheader:
// CHECK: br label %omp_iterator.header
//
// Header has the IV phi and branches to cond
// CHECK: omp_iterator.header:
// CHECK: [[IV:%.*]] = phi i64 [ 0, %omp_iterator.preheader ], [ [[NEXT:%.*]], %omp_iterator.inc ]
// CHECK: br label %omp_iterator.cond
//
// Cond: IV < 48 and branches to body or exit
// CHECK: omp_iterator.cond:
// CHECK: [[CMP:%.*]] = icmp ult i64 [[IV]], 48
// CHECK: br i1 [[CMP]], label %omp_iterator.body, label %omp_iterator.exit
//
// Exit -> After -> continuation
// CHECK: omp_iterator.exit:
// CHECK: br label %omp_iterator.after
// CHECK: omp_iterator.after:
// CHECK: br label %omp.it.cont
//
// Body: store into affinity_list[IV] then branch to inc
// CHECK: omp_iterator.body:
// CHECK: [[ENTRY:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr %loadgep_omp.affinity_list, i64 [[IV]]
// CHECK: [[ADDRI64:%.*]] = ptrtoint ptr %loadgep_ to i64
// CHECK: [[ADDRGEP:%.*]] = getelementptr inbounds nuw { i64, i64, i32 }, ptr [[ENTRY]], i32 0, i32 0
// CHECK: store i64 [[ADDRI64]], ptr [[ADDRGEP]]
// CHECK: [[LENGEP:%.*]] = getelementptr inbounds nuw { i64, i64, i32 }, ptr [[ENTRY]], i32 0, i32 1
// CHECK: store i64 4, ptr [[LENGEP]]
// CHECK: [[FLAGGEP:%.*]] = getelementptr inbounds nuw { i64, i64, i32 }, ptr [[ENTRY]], i32 0, i32 2
// CHECK: store i32 0, ptr [[FLAGGEP]]
// CHECK: br label %omp_iterator.inc
//
// CHECK: omp_iterator.inc:
// CHECK: [[NEXT]] = add nuw i64 [[IV]], 1
// CHECK: br label %omp_iterator.header

llvm.func @task_affinity_iterator_multiple(%arr: !llvm.ptr {llvm.nocapture}) {
  %c1  = llvm.mlir.constant(1 : i64) : i64
  %c3  = llvm.mlir.constant(3 : i64) : i64
  %c4  = llvm.mlir.constant(4 : i64) : i64
  %c6  = llvm.mlir.constant(6 : i64) : i64
  %len = llvm.mlir.constant(4 : i64) : i64

  omp.parallel {
    omp.single {
      // First iterator: 2-D (4 * 6 = 24)
      %it0 = omp.iterator(%i: i64, %j: i64) =
          (%c1 to %c4 step %c1, %c1 to %c6 step %c1) {
        %entry0 = omp.affinity_entry %arr, %len
            : (!llvm.ptr, i64) -> !omp.affinity_entry_ty<!llvm.ptr, i64>
        omp.yield(%entry0 : !omp.affinity_entry_ty<!llvm.ptr, i64>)
      } -> !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>

      // second iterator: 1-D (3)
      %it1 = omp.iterator(%k: i64) = (%c1 to %c3 step %c1) {
        %entry1 = omp.affinity_entry %arr, %len
            : (!llvm.ptr, i64) -> !omp.affinity_entry_ty<!llvm.ptr, i64>
        omp.yield(%entry1 : !omp.affinity_entry_ty<!llvm.ptr, i64>)
      } -> !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>

      // Multiple iterators in a single affinity clause.
      omp.task affinity(%it0: !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>,
            %it1: !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>) {
        omp.terminator
      }

      omp.terminator
    }
    omp.terminator
  }

  llvm.return
}

// CHECK-LABEL: define internal void @task_affinity_iterator_multiple..omp_par.3(
// CHECK-DAG: %gep_omp.affinity_list = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 0
// CHECK-DAG: %gep_omp.affinity_list{{.*}} = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 1

// First iterator header
// CHECK: omp_iterator.preheader:
// CHECK: br label %[[HEADER0:.+]]
// CHECK: [[HEADER0]]:
// CHECK: [[IV0:%.*]] = phi i64 [ 0, %omp_iterator.preheader ], [ [[NEXT0:%.*]], %[[INC0:.+]] ]
// CHECK: br label %[[COND0:.+]]
// CHECK: [[COND0]]:
// CHECK: [[CMP0:%.*]] = icmp ult i64 [[IV0]], 24
// CHECK: br i1 [[CMP0]], label %[[BODY0:.+]], label %omp_iterator.exit

// Second iterator header
// CHECK: omp_iterator.preheader{{.*}}:
// CHECK: [[HEADER1:.+]]:
// CHECK: [[IV1:%.*]] = phi i64 [ 0, %omp_iterator.preheader{{.*}} ], [ [[NEXT1:%.*]], %[[INC1:.+]] ]
// CHECK: br label %omp_iterator.cond{{.*}}
// CHECK: omp_iterator.cond{{.*}}:
// CHECK: [[CMP1:%.*]] = icmp ult i64 [[IV1]], 3
// CHECK: br i1 [[CMP1]], label %[[BODY1:.+]], label %omp_iterator.exit{{.*}}

// CHECK: codeRepl:
// CHECK: call ptr @__kmpc_omp_task_alloc
// CHECK: call i32 @__kmpc_omp_reg_task_with_affinity{{.*}}i32 24{{.*}}ptr %loadgep_omp.affinity_list
// CHECK: call i32 @__kmpc_omp_reg_task_with_affinity{{.*}}i32 3{{.*}}ptr %loadgep_omp.affinity_list{{.*}}
// CHECK: call i32 @__kmpc_omp_task

// Second iterator body
// CHECK: [[BODY1]]:
// CHECK: [[ENTRY1:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr %loadgep_omp.affinity_list
// CHECK: [[ADDR1:%.*]] = ptrtoint ptr %loadgep_ to i64
// CHECK: [[ADDRGEP1:%.*]] = getelementptr inbounds{{.*}} { i64, i64, i32 }, ptr [[ENTRY1]], i32 0, i32 0
// CHECK: store i64 [[ADDR1]], ptr [[ADDRGEP1]]
// CHECK: [[LENGEP1:%.*]] = getelementptr inbounds{{.*}} { i64, i64, i32 }, ptr [[ENTRY1]], i32 0, i32 1
// CHECK: store i64 4, ptr [[LENGEP1]]
// CHECK: [[FLAGGEP1:%.*]] = getelementptr inbounds{{.*}} { i64, i64, i32 }, ptr [[ENTRY1]], i32 0, i32 2
// CHECK: store i32 0, ptr [[FLAGGEP1]]
// CHECK: br label %[[INC1]]
// CHECK: [[INC1]]:
// CHECK: [[NEXT1]] = add nuw i64 [[IV1]], 1
// CHECK: br label %[[HEADER1]]

// First iterator body
// CHECK: [[BODY0]]:
// CHECK: [[ENTRY0:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr %loadgep_omp.affinity_list, i64 [[IV0]]
// CHECK: [[ADDR0:%.*]] = ptrtoint ptr %loadgep_ to i64
// CHECK: [[ADDRGEP0:%.*]] = getelementptr inbounds{{.*}} { i64, i64, i32 }, ptr [[ENTRY0]], i32 0, i32 0
// CHECK: store i64 [[ADDR0]], ptr [[ADDRGEP0]]
// CHECK: [[LENGEP0:%.*]] = getelementptr inbounds{{.*}} { i64, i64, i32 }, ptr [[ENTRY0]], i32 0, i32 1
// CHECK: store i64 4, ptr [[LENGEP0]]
// CHECK: [[FLAGGEP0:%.*]] = getelementptr inbounds{{.*}} { i64, i64, i32 }, ptr [[ENTRY0]], i32 0, i32 2
// CHECK: store i32 0, ptr [[FLAGGEP0]]
// CHECK: br label %[[INC0]]
// CHECK: [[INC0]]:
// CHECK: [[NEXT0]] = add nuw i64 [[IV0]], 1
// CHECK: br label %[[HEADER0]]
