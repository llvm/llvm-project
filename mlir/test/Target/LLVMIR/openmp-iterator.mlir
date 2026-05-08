// RUN: split-file %s %t
// RUN: mlir-translate --mlir-to-llvmir %t/host.mlir | FileCheck %s --check-prefix=CHECK
// RUN: mlir-translate --mlir-to-llvmir %t/target.mlir | FileCheck %s --check-prefix=TARGET

//--- host.mlir

// --------------------------------------------------------------------
// Affinity clause
// --------------------------------------------------------------------

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
// CHECK: [[ENTRY:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr %{{.*affinity_list.*}}, i64 [[IV]]
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
// CHECK: [[ENTRY:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr %{{.*affinity_list.*}}, i64 [[IV]]
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

// CHECK-LABEL: define internal void @task_affinity_iterator_multiple
// CHECK-DAG: [[AFFLIST0:%.*]] = alloca { i64, i64, i32 }, i64 24, align 8
// CHECK-DAG: [[AFFLIST1:%.*]] = alloca { i64, i64, i32 }, i64 3, align 8
// CHECK-DAG: [[AFFINITY_LIST:%.*]] = alloca { i64, i64, i32 }, i32 27, align 8

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

// CHECK: [[AFFINITY_LIST_1:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr [[AFFINITY_LIST]], i64 0
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 [[AFFINITY_LIST_1]], ptr align 1 [[AFFLIST0]], i64 480, i1 false)
// CHECK: [[AFFINITY_LIST_2:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr [[AFFINITY_LIST]], i64 24
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 [[AFFINITY_LIST_2]], ptr align 1 [[AFFLIST1]], i64 60, i1 false)
// CHECK: codeRepl:
// CHECK: call ptr @__kmpc_omp_task_alloc
// CHECK: call i32 @__kmpc_omp_reg_task_with_affinity{{.*}}i32 27{{.*}}ptr [[AFFINITY_LIST]]
// CHECK: call i32 @__kmpc_omp_task

// Second iterator body
// CHECK: [[BODY1]]:
// CHECK: [[ENTRY1:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr [[AFFLIST1]]
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
// CHECK: [[ENTRY0:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr [[AFFLIST0]], i64 [[IV0]]
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

// Makes sure affinity list only created after dynamic count
llvm.func @task_affinity_iterator_dynamic_tripcount(
    %arr: !llvm.ptr {llvm.nocapture}, %lb: i64, %ub: i64, %step: i64,
    %len: i64) {
  omp.parallel {
    omp.single {
      %it = omp.iterator(%i: i64) = (%lb to %ub step %step) {
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

// CHECK-LABEL: define internal void @task_affinity_iterator_dynamic_tripcount
// CHECK: [[DIFF:%.*]] = sub i64 {{.*}}, {{.*}}
// CHECK: [[DIV:%.*]] = sdiv i64 [[DIFF]], {{.*}}
// CHECK: [[TRIPS:%.*]] = add i64 [[DIV]], 1
// CHECK: [[SCALED:%.*]] = mul i64 1, [[TRIPS]]
// CHECK: [[AFFLIST:%.*]] = alloca { i64, i64, i32 }, i64 [[SCALED]]

llvm.func @task_affinity_iterator_negative_step(%arr: !llvm.ptr {llvm.nocapture}) {
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %cn1 = llvm.mlir.constant(-1 : i64) : i64

  omp.parallel {
    omp.single {
      %it = omp.iterator(%i: i64) = (%c4 to %c1 step %cn1) {
        %entry = omp.affinity_entry %arr, %i
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

// CHECK-LABEL: define internal void @task_affinity_iterator_negative_step
// CHECK: [[AFFLIST:%.*]] = alloca { i64, i64, i32 }, i64 4, align 8
// CHECK: omp_iterator.cond:
// CHECK: [[CMP:%.*]] = icmp ult i64 %omp_iterator.iv, 4
// CHECK: br i1 [[CMP]], label %omp_iterator.body, label %omp_iterator.exit
// CHECK: omp_iterator.body:
// CHECK: [[IDX:%.*]] = urem i64 %omp_iterator.iv, 4
// CHECK: [[STEPMUL:%.*]] = mul i64 [[IDX]], -1
// CHECK: [[PHYSIV:%.*]] = add i64 4, [[STEPMUL]]
// CHECK: [[ENTRY:%.*]] = getelementptr inbounds { i64, i64, i32 }, ptr [[AFFLIST]], i64 %omp_iterator.iv
// CHECK: [[LENPTR:%.*]] = getelementptr inbounds nuw { i64, i64, i32 }, ptr [[ENTRY]], i32 0, i32 1
// CHECK: store i64 [[PHYSIV]], ptr [[LENPTR]]

// --------------------------------------------------------------------
// Depend clause
// --------------------------------------------------------------------

llvm.func @omp_task_depend_iterator_simple(%addr : !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c10 = llvm.mlir.constant(10 : i64) : i64
  %step = llvm.mlir.constant(1 : i64) : i64

  %it = omp.iterator(%iv: i64) = (%c1 to %c10 step %step) {
    omp.yield(%addr : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>

  omp.task depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @omp_task_depend_iterator_simple
// CHECK-SAME: (ptr %[[ADDR:[0-9]+]])
// CHECK: %[[DEP_ARR:.*]] = tail call ptr @malloc(i64 %mallocsize)
//
// Iterator loop: preheader -> header -> cond -> body -> inc -> header...
// CHECK: omp_dep_iterator.header:
// CHECK: %[[IV:.*]] = phi i64 [ 0, %omp_dep_iterator.preheader ], [ %[[NEXT:.*]], %omp_dep_iterator.inc ]
// CHECK: omp_dep_iterator.cond:
// CHECK: %[[CMP:.*]] = icmp ult i64 %[[IV]], 10
// CHECK: br i1 %[[CMP]], label %omp_dep_iterator.body, label %omp_dep_iterator.exit
//
// Body: store kmp_dep_info at depArray[0 + linearIV]
// CHECK: omp_dep_iterator.body:
// CHECK: %[[IDX:.*]] = add i64 0, %[[IV]]
// CHECK: %[[ENTRY:.*]] = getelementptr inbounds %struct.kmp_dep_info, ptr %[[DEP_ARR]], i64 %[[IDX]]
// CHECK: %[[BASE_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[ENTRY]], i32 0, i32 0
// CHECK: %[[PTRINT:.*]] = ptrtoint ptr %[[ADDR]] to i64
// CHECK: store i64 %[[PTRINT]], ptr %[[BASE_GEP]]
// CHECK: %[[LEN_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[ENTRY]], i32 0, i32 1
// CHECK: store i64 8, ptr %[[LEN_GEP]]
// CHECK: %[[FLAGS_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[ENTRY]], i32 0, i32 2
// depKind = 1 (DepIn)
// CHECK: store i8 1, ptr %[[FLAGS_GEP]]
//
// CHECK: omp_dep_iterator.inc:
// CHECK: %[[NEXT]] = add nuw i64 %[[IV]], 1
//
// Task creation with deps, then free
// CHECK: call i32 @__kmpc_omp_task_with_deps(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 10, ptr %[[DEP_ARR]], i32 0, ptr null)
// CHECK: tail call void @free(ptr %[[DEP_ARR]])

llvm.func @omp_task_depend_iterator_mixed(%addr : !llvm.ptr, %plain : !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c10 = llvm.mlir.constant(10 : i64) : i64
  %step = llvm.mlir.constant(1 : i64) : i64

  %it = omp.iterator(%iv: i64) = (%c1 to %c10 step %step) {
    omp.yield(%addr : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>

  omp.task depend(taskdependout -> %plain : !llvm.ptr, taskdependin -> %it : !omp.iterated<!llvm.ptr>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @omp_task_depend_iterator_mixed
// CHECK-SAME: (ptr %[[ADDR2:[0-9]+]], ptr %[[PLAIN:[0-9]+]])
// CHECK: %[[DEP_ARR2:.*]] = tail call ptr @malloc(i64 %mallocsize)
//
// Plain entry at index 0
// CHECK: %[[PLAIN_ENTRY:.*]] = getelementptr inbounds %struct.kmp_dep_info, ptr %[[DEP_ARR2]], i64 0
// CHECK: getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[PLAIN_ENTRY]], i32 0, i32 0
// CHECK: %[[PLAIN_PTRINT:.*]] = ptrtoint ptr %[[PLAIN]] to i64
// CHECK: store i64 %[[PLAIN_PTRINT]], ptr
// depKind = 3 (DepInOut/out)
// CHECK: store i8 3, ptr
//
// Iterator loop for iterated entry starting at offset 1
// CHECK: omp_dep_iterator.body:
// startIdx(1) + linearIV
// CHECK: add i64 1, %omp_dep_iterator.iv
// CHECK: getelementptr inbounds %struct.kmp_dep_info, ptr %[[DEP_ARR2]]
// depKind = 1 (DepIn)
// CHECK: store i8 1, ptr
//
// CHECK: call i32 @__kmpc_omp_task_with_deps(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 11, ptr %[[DEP_ARR2]], i32 0, ptr null)
// CHECK: tail call void @free(ptr %[[DEP_ARR2]])

// Dynamic bounds: iterator bounds are function arguments, so the trip count
// and dep-array size are computed at runtime.  The alloca must be placed
// after the trip-count computation (not hoisted to the entry block)
// to avoid "instruction does not dominate all uses" errors.
llvm.func @omp_task_depend_iterator_dynamic(%addr : !llvm.ptr,
    %lb : i64, %ub : i64, %step : i64) {
  %it = omp.iterator(%iv: i64) = (%lb to %ub step %step) {
    omp.yield(%addr : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>

  omp.task depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @omp_task_depend_iterator_dynamic
//
// Tripcount computation from dynamic bounds
// CHECK: %[[DIFF:.*]] = sub i64 %{{.*}}, %{{.*}}
// CHECK: %[[DIV:.*]] = sdiv i64 %[[DIFF]], %{{.*}}
// CHECK: %[[TRIPS:.*]] = add i64 %[[DIV]], 1
// CHECK: %[[SCALED:.*]] = mul i64 1, %[[TRIPS]]
// Dynamic total = 0 + scaled trip count
// CHECK: %[[TOTAL:.*]] = add i64 0, %[[SCALED]]
//
// Malloc with dynamic size
// CHECK: %[[DEP_ARR:.*]] = tail call ptr @malloc(i64 %mallocsize)
// CHECK: omp_dep_iterator.body:
// CHECK: getelementptr inbounds %struct.kmp_dep_info, ptr %[[DEP_ARR]]
// NumDeps is truncated to i32 for the runtime call
// CHECK: %[[NDEPS:.*]] = trunc i64 %[[TOTAL]] to i32
// CHECK: call i32 @__kmpc_omp_task_with_deps(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 %[[NDEPS]], ptr %[[DEP_ARR]], i32 0, ptr null)
// CHECK: tail call void @free(ptr %[[DEP_ARR]])

// Dynamic bounds with mixed plain + iterated depends.
llvm.func @omp_task_depend_iterator_dynamic_mixed(%addr : !llvm.ptr,
    %plain : !llvm.ptr, %lb : i64, %ub : i64, %step : i64) {
  %it = omp.iterator(%iv: i64) = (%lb to %ub step %step) {
    omp.yield(%addr : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>

  omp.task depend(taskdependout -> %plain : !llvm.ptr, taskdependin -> %it : !omp.iterated<!llvm.ptr>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @omp_task_depend_iterator_dynamic_mixed
// CHECK: %[[TRIPS2:.*]] = mul i64 1, %{{.*}}
// total = 1 (plain) + dynamic trip count
// CHECK: %[[TOTAL2:.*]] = add i64 1, %[[TRIPS2]]
// CHECK: %[[DEP_ARR2:.*]] = tail call ptr @malloc(i64 %mallocsize)
// Plain entry at index 0
// CHECK: %[[PLAIN_ENTRY:.*]] = getelementptr inbounds %struct.kmp_dep_info, ptr %[[DEP_ARR2]], i64 0
// CHECK: store i8 3, ptr
// Iterator loop
// CHECK: omp_dep_iterator.body:
// CHECK: add i64 1, %omp_dep_iterator.iv
// CHECK: %[[NDEPS2:.*]] = trunc i64 %[[TOTAL2]] to i32
// CHECK: call i32 @__kmpc_omp_task_with_deps(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 %[[NDEPS2]], ptr %[[DEP_ARR2]], i32 0, ptr null)
// CHECK: tail call void @free(ptr %[[DEP_ARR2]])

//--- target.mlir

// --------------------------------------------------------------------
// Depend clause on target construct
// --------------------------------------------------------------------

// Target construct with iterator-based depend clause.
// The iterator(i=1:10) should allocate a kmp_dep_info[10] array, fill it via
// a dep_iterator loop, then emit __kmpc_omp_wait_deps with ndeps=10.
module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @omp_target_depend_iterator(%addr: !llvm.ptr) {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c10 = llvm.mlir.constant(10 : i64) : i64
    %step = llvm.mlir.constant(1 : i64) : i64

    %it = omp.iterator(%iv: i64) = (%c1 to %c10 step %step) {
      omp.yield(%addr : !llvm.ptr)
    } -> !omp.iterated<!llvm.ptr>

    %map = omp.map.info var_ptr(%addr : !llvm.ptr, i32) map_clauses(to) capture(ByRef) -> !llvm.ptr {name = "data"}
    omp.target depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>) map_entries(%map -> %arg0 : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// TARGET-LABEL: define void @omp_target_depend_iterator
// TARGET-SAME: (ptr %[[ADDR:[0-9]+]])
// TARGET-DAG: %[[DEP_ARR:.*]] = tail call ptr @malloc(i64 %mallocsize)
//
// Iterator loop: preheader -> header -> cond -> body -> inc -> header...
// TARGET: omp_dep_iterator.header:
// TARGET: %[[IV:.*]] = phi i64 [ 0, %omp_dep_iterator.preheader ], [ %[[NEXT:.*]], %omp_dep_iterator.inc ]
// TARGET: omp_dep_iterator.cond:
// TARGET: %[[CMP:.*]] = icmp ult i64 %[[IV]], 10
// TARGET: br i1 %[[CMP]], label %omp_dep_iterator.body, label %omp_dep_iterator.exit
//
// Body: store kmp_dep_info at depArray[0 + linearIV]
// TARGET: omp_dep_iterator.body:
// TARGET: %[[IDX:.*]] = add i64 0, %[[IV]]
// TARGET: %[[ENTRY:.*]] = getelementptr inbounds %struct.kmp_dep_info, ptr %[[DEP_ARR]], i64 %[[IDX]]
// TARGET: %[[BASE_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[ENTRY]], i32 0, i32 0
// TARGET: %[[PTRINT:.*]] = ptrtoint ptr %[[ADDR]] to i64
// TARGET: store i64 %[[PTRINT]], ptr %[[BASE_GEP]]
// TARGET: %[[LEN_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[ENTRY]], i32 0, i32 1
// TARGET: store i64 8, ptr %[[LEN_GEP]]
// TARGET: %[[FLAGS_GEP:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[ENTRY]], i32 0, i32 2
// depKind = 1 (DepIn)
// TARGET: store i8 1, ptr %[[FLAGS_GEP]]
//
// TARGET: omp_dep_iterator.inc:
// TARGET: %[[NEXT]] = add nuw i64 %[[IV]], 1
//
// Target task: wait_deps with ndeps=10, then begin_if0/proxy/complete_if0, then free
// TARGET: call void @__kmpc_omp_wait_deps(ptr @{{.*}}, i32 %{{.*}}, i32 10, ptr %[[DEP_ARR]], i32 0, ptr null)
// TARGET: call void @__kmpc_omp_task_begin_if0
// TARGET: call void @.omp_target_task_proxy_func
// TARGET: call void @__kmpc_omp_task_complete_if0
// TARGET: tail call void @free(ptr %[[DEP_ARR]])
