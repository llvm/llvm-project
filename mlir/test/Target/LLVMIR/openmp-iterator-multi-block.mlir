// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// An omp.iterator region may span several blocks. Each entry stores the value
// yielded by the region's single omp.yield.

// The yielded address is selected by a branch.
llvm.func @depend_diamond(%a : !llvm.ptr, %b : !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %it = omp.iterator(%iv: i64) = (%c1 to %c4 step %c1) {
    %first = llvm.icmp "eq" %iv, %c1 : i64
    llvm.cond_br %first, ^bb1(%a : !llvm.ptr), ^bb1(%b : !llvm.ptr)
  ^bb1(%p : !llvm.ptr):
    omp.yield(%p : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.task depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @depend_diamond
// CHECK-SAME: (ptr %[[A:[0-9]+]], ptr %[[B:[0-9]+]])
// CHECK: %[[DEPS:.*]] = tail call ptr @malloc(i64 80)
// CHECK: omp_dep_iterator.body:
// CHECK: %[[IV:.*]] = add i64 1, %{{.*}}
// CHECK: br label %[[ENTRY:omp.iterator.region]]
// CHECK: [[ENTRY]]:
// CHECK: %[[FIRST:.*]] = icmp eq i64 %[[IV]], 1
// CHECK: br i1 %[[FIRST]], label %[[MERGE:omp.iterator.region[0-9]+]], label %[[ELSE:omp.iterator.region[0-9]+]]
// CHECK: [[MERGE]]:
// CHECK: %[[P:.*]] = phi ptr [ %[[BV:.*]], %[[ELSE]] ], [ %[[A]], %[[ENTRY]] ]
// CHECK: br label %omp.iterator.region.cont
// CHECK: [[ELSE]]:
// CHECK: %[[BV]] = phi ptr [ %[[B]], %[[ENTRY]] ]
// CHECK: omp.iterator.region.cont:
// CHECK: %[[ADDR:.*]] = ptrtoint ptr %[[P]] to i64
// CHECK: store i64 %[[ADDR]], ptr
// CHECK: br label %omp_dep_iterator.inc
// CHECK: call i32 @__kmpc_omp_task_with_deps({{.*}}, i32 4, ptr %[[DEPS]], i32 0, ptr null)

// The affinity length is defined before the branch, the address after it.
llvm.func @affinity_split(%a : !llvm.ptr, %b : !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %it = omp.iterator(%iv: i64) = (%c1 to %c3 step %c1) {
    %len = llvm.mul %iv, %c3 : i64
    %first = llvm.icmp "eq" %iv, %c1 : i64
    llvm.cond_br %first, ^bb1, ^bb2
  ^bb1:
    llvm.br ^bb3(%a : !llvm.ptr)
  ^bb2:
    llvm.br ^bb3(%b : !llvm.ptr)
  ^bb3(%p : !llvm.ptr):
    %e = omp.affinity_entry %p, %len
        : (!llvm.ptr, i64) -> !omp.affinity_entry_ty<!llvm.ptr, i64>
    omp.yield(%e : !omp.affinity_entry_ty<!llvm.ptr, i64>)
  } -> !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>
  omp.task affinity(%it : !omp.iterated<!omp.affinity_entry_ty<!llvm.ptr, i64>>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @affinity_split
// CHECK-SAME: (ptr %[[A:[0-9]+]], ptr %[[B:[0-9]+]])
// CHECK: omp_iterator.body:
// CHECK: %[[IV:.*]] = add i64 1, %{{.*}}
// CHECK: omp.iterator.region:
// CHECK: %[[LEN:.*]] = mul i64 %[[IV]], 3
// CHECK: %[[P:.*]] = phi ptr [ %[[B]], %{{.*}} ], [ %[[A]], %{{.*}} ]
// CHECK: omp.iterator.region.cont:
// CHECK: %[[ADDR:.*]] = ptrtoint ptr %[[P]] to i64
// CHECK: store i64 %[[ADDR]], ptr
// CHECK: store i64 %[[LEN]], ptr
// CHECK: br label %omp_iterator.inc
// CHECK: call i32 @__kmpc_omp_reg_task_with_affinity({{.*}}, i32 3, ptr

// A loop inside the region computes the yielded address.
llvm.func @depend_inner_loop(%addr : !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %it = omp.iterator(%iv: i64) = (%c0 to %c3 step %c1) {
    llvm.br ^loop(%c0, %c0 : i64, i64)
  ^loop(%k : i64, %acc : i64):
    %done = llvm.icmp "sge" %k, %iv : i64
    llvm.cond_br %done, ^exit, ^body
  ^body:
    %acc2 = llvm.add %acc, %k : i64
    %k2 = llvm.add %k, %c1 : i64
    llvm.br ^loop(%k2, %acc2 : i64, i64)
  ^exit:
    %p = llvm.getelementptr %addr[%acc] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    omp.yield(%p : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.task depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @depend_inner_loop
// CHECK-SAME: (ptr %[[ADDR:[0-9]+]])
// CHECK: omp_dep_iterator.body:
// CHECK: %[[IV:.*]] = add i64 0, %{{.*}}
// CHECK: [[LOOP:omp.iterator.region[0-9]+]]:
// CHECK: %[[K:.*]] = phi i64 [ %[[K2:.*]], %[[BODY:omp.iterator.region[0-9]+]] ], [ 0, %omp.iterator.region ]
// CHECK: %[[ACC:.*]] = phi i64 [ %[[ACC2:.*]], %[[BODY]] ], [ 0, %omp.iterator.region ]
// CHECK: %[[DONE:.*]] = icmp sge i64 %[[K]], %[[IV]]
// CHECK: br i1 %[[DONE]], label %[[EXIT:omp.iterator.region[0-9]+]], label %[[BODY]]
// CHECK: [[BODY]]:
// CHECK: %[[ACC2]] = add i64 %[[ACC]], %[[K]]
// CHECK: %[[K2]] = add i64 %[[K]], 1
// CHECK: br label %[[LOOP]]
// CHECK: [[EXIT]]:
// CHECK: %[[P:.*]] = getelementptr i8, ptr %[[ADDR]], i64 %[[ACC]]
// CHECK: omp.iterator.region.cont:
// CHECK: %[[PI:.*]] = ptrtoint ptr %[[P]] to i64
// CHECK: store i64 %[[PI]], ptr
// CHECK: call i32 @__kmpc_omp_task_with_deps({{.*}}, i32 4, ptr

// Both induction values are mapped in their declared order.
llvm.func @depend_2d(%a : !llvm.ptr, %b : !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %it = omp.iterator(%i: i64, %j: i64) = (%c1 to %c2 step %c1, %c1 to %c3 step %c1) {
    %lt = llvm.icmp "slt" %i, %j : i64
    llvm.cond_br %lt, ^bb1(%a : !llvm.ptr), ^bb1(%b : !llvm.ptr)
  ^bb1(%p : !llvm.ptr):
    omp.yield(%p : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.task depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>) {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @depend_2d
// CHECK-SAME: (ptr %[[A:[0-9]+]], ptr %[[B:[0-9]+]])
// CHECK: omp_dep_iterator.body:
// CHECK: %[[J:.*]] = add i64 1, %{{.*}}
// CHECK: %[[I:.*]] = add i64 1, %{{.*}}
// CHECK: omp.iterator.region:
// CHECK: icmp slt i64 %[[I]], %[[J]]
// CHECK: %[[P:.*]] = phi ptr [ %{{.*}}, %{{.*}} ], [ %[[A]], %omp.iterator.region ]
// CHECK: phi ptr [ %[[B]], %omp.iterator.region ]
// CHECK: omp.iterator.region.cont:
// CHECK: %[[PI:.*]] = ptrtoint ptr %[[P]] to i64
// CHECK: store i64 %[[PI]], ptr
// CHECK: call i32 @__kmpc_omp_task_with_deps({{.*}}, i32 6, ptr

// A block argument fed by the induction value keeps its declared type.
llvm.func @depend_narrow(%x : !llvm.ptr) {
  %lo = llvm.mlir.constant(1 : i32) : i32
  %hi = llvm.mlir.constant(2 : i32) : i32
  %it = omp.iterator(%i: i32) = (%lo to %hi step %lo) {
    llvm.br ^next(%i : i32)
  ^next(%v: i32):
    llvm.store %v, %x : i32, !llvm.ptr
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.taskwait depend(taskdependin -> %it : !omp.iterated<!llvm.ptr>)
  llvm.return
}

// CHECK-LABEL: define void @depend_narrow
// CHECK: omp_dep_iterator.body:
// CHECK: %[[IDX:.*]] = trunc i64 %{{.*}} to i32
// CHECK: %[[OFF:.*]] = mul i32 %[[IDX]], 1
// CHECK: %[[IV:.*]] = add i32 1, %[[OFF]]
// CHECK: %[[V:.*]] = phi i32 [ %[[IV]], %omp.iterator.region ]
// CHECK: store i32 %[[V]], ptr
