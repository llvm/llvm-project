// RUN: mlir-translate --mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @taskwait_depend(%x: !llvm.ptr) {
  omp.taskwait depend(taskdependout -> %x : !llvm.ptr)
  llvm.return
}

// CHECK-LABEL: define void @taskwait_depend
// CHECK-SAME: (ptr[[xaddr:.+]])
// CHECK: %[[dep_arr_addr:.+]] = alloca [1 x %struct.kmp_dep_info], align 8
// CHECK: %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num({{.+}})
// CHECK: call void @__kmpc_omp_taskwait_deps_51(ptr @{{.+}}, i32 %[[omp_global_thread_num]], i32 1, ptr %[[dep_arr_addr]], i32 0, ptr null, i32 0)

llvm.func @taskwait_depend_iterator(%x: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c10 = llvm.mlir.constant(10 : i64) : i64
  %step = llvm.mlir.constant(1 : i64) : i64

  %ix = omp.iterator(%i: i64) = (%c1 to %c10 step %step) {
    omp.yield(%x : !llvm.ptr)
  } -> !omp.iterated<!llvm.ptr>
  omp.taskwait depend(taskdependin -> %ix : !omp.iterated<!llvm.ptr>)
  llvm.return
}

// CHECK-LABEL: define void @taskwait_depend_iterator
// CHECK-SAME: (ptr[[xaddr:.+]])
// CHECK: %[[dep_arr_addr:.+]] = tail call ptr @malloc(i64 %mallocsize)
//
// CHECK: omp_dep_iterator.header:
// CHECK: %[[iv:.*]] = phi i64 [ 0, %omp_dep_iterator.preheader ], [ %[[next:.*]], %omp_dep_iterator.inc ]
//
// CHECK: omp_dep_iterator.cond:
// CHECK: %[[cmp:.*]] = icmp ult i64 %[[iv]], 10
// CHECK: br i1 %[[cmp]], label %omp_dep_iterator.body, label %omp_dep_iterator.exit
//
// CHECK: omp_dep_iterator.body:
// CHECK: %[[idx:.*]] = add i64 0, %[[iv]]
// CHECK: %[[entry:.*]] = getelementptr inbounds %struct.kmp_dep_info, ptr %[[dep_arr_addr]], i64 %[[idx]]
// CHECK: %[[base_gep:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[entry]], i32 0, i32 0
// CHECK: %[[ptrint:.*]] = ptrtoint ptr[[xaddr]] to i64
// CHECK: store i64 %[[ptrint]], ptr %[[base_gep]]
// CHECK: %[[len_gep:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[entry]], i32 0, i32 1
// CHECK: store i64 8, ptr %[[len_gep]]
// CHECK: %[[flags_gep:.*]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[entry]], i32 0, i32 2
// CHECK: store i8 1, ptr %[[flags_gep]]
//
// CHECK: omp_dep_iterator.inc:
// CHECK: %[[next]] = add nuw i64 %[[iv]], 1
//
// CHECK: %[[omp_global_thread_num:.+]] = call i32 @__kmpc_global_thread_num({{.+}})
// CHECK: call void @__kmpc_omp_taskwait_deps_51(ptr @{{.+}}, i32 %[[omp_global_thread_num]], i32 10, ptr %[[dep_arr_addr]], i32 0, ptr null, i32 0)
// CHECK: tail call void @free(ptr %[[dep_arr_addr:.+]])
