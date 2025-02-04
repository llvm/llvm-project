// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @_QQmain() attributes {fir.bindc_name = "main"} {
    %0 = llvm.mlir.constant(39 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(40 : index) : i64
    %4 = llvm.mlir.addressof @_QFEa : !llvm.ptr
    %5 = llvm.mlir.addressof @_QFEb : !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
    %8 = llvm.mlir.addressof @_QFEn : !llvm.ptr
    omp.task {
      %14 = llvm.mlir.constant(1 : i64) : i64
      %15 = llvm.alloca %14 x i32 {bindc_name = "i", pinned} : (i64) -> !llvm.ptr
      %16 = llvm.load %8 : !llvm.ptr -> i32
      %17 = llvm.sext %16 : i32 to i64
      %18 = llvm.trunc %2 : i64 to i32
      llvm.br ^bb1(%18, %17 : i32, i64)
    ^bb1(%19: i32, %20: i64):  // 2 preds: ^bb0, ^bb2
      %21 = llvm.icmp "sgt" %20, %1 : i64
      llvm.cond_br %21, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      llvm.store %19, %15 : i32, !llvm.ptr
      %22 = llvm.load %15 : !llvm.ptr -> i32
      %23 = llvm.sext %22 : i32 to i64
      %24 = llvm.mlir.constant(1 : i64) : i64
      %25 = llvm.mlir.constant(0 : i64) : i64
      %26 = llvm.sub %23, %24 overflow<nsw> : i64
      %27 = llvm.mul %26, %24 overflow<nsw> : i64
      %28 = llvm.mul %27, %24 overflow<nsw> : i64
      %29 = llvm.add %28, %25 overflow<nsw> : i64
      %30 = llvm.mul %24, %3 overflow<nsw> : i64
      %31 = llvm.getelementptr %4[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %22, %31 : i32, !llvm.ptr
      %32 = llvm.load %15 : !llvm.ptr -> i32
      %33 = llvm.add %32, %18 : i32
      %34 = llvm.sub %20, %2 : i64
      llvm.br ^bb1(%33, %34 : i32, i64)
    ^bb3:  // pred: ^bb1
      llvm.store %19, %15 : i32, !llvm.ptr
      omp.terminator
    }
    %9 = omp.map.bounds lower_bound(%1 : i64) upper_bound(%0 : i64) extent(%3 : i64) stride(%2 : i64) start_idx(%2 : i64)
    %10 = omp.map.info var_ptr(%4 : !llvm.ptr, !llvm.array<40 x i32>) map_clauses(to) capture(ByRef) bounds(%9) -> !llvm.ptr {name = "a"}
    %11 = omp.map.info var_ptr(%5 : !llvm.ptr, !llvm.array<40 x i32>) map_clauses(from) capture(ByRef) bounds(%9) -> !llvm.ptr {name = "b"}
    %12 = omp.map.info var_ptr(%7 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "i"}
    %13 = omp.map.info var_ptr(%8 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "n"}
    omp.target depend(taskdependin -> %4 : !llvm.ptr) map_entries(%10 -> %arg0, %11 -> %arg1, %12 -> %arg2, %13 -> %arg3 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %14 = llvm.mlir.constant(0 : index) : i64
      %15 = llvm.mlir.constant(10 : i32) : i32
      %16 = llvm.mlir.constant(1 : index) : i64
      %17 = llvm.mlir.constant(40 : index) : i64
      %18 = llvm.load %arg3 : !llvm.ptr -> i32
      %19 = llvm.sext %18 : i32 to i64
      %20 = llvm.trunc %16 : i64 to i32
      llvm.br ^bb1(%20, %19 : i32, i64)
    ^bb1(%21: i32, %22: i64):  // 2 preds: ^bb0, ^bb2
      %23 = llvm.icmp "sgt" %22, %14 : i64
      llvm.cond_br %23, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      llvm.store %21, %arg2 : i32, !llvm.ptr
      %24 = llvm.load %arg2 : !llvm.ptr -> i32
      %25 = llvm.sext %24 : i32 to i64
      %26 = llvm.mlir.constant(1 : i64) : i64
      %27 = llvm.mlir.constant(0 : i64) : i64
      %28 = llvm.sub %25, %26 overflow<nsw> : i64
      %29 = llvm.mul %28, %26 overflow<nsw> : i64
      %30 = llvm.mul %29, %26 overflow<nsw> : i64
      %31 = llvm.add %30, %27 overflow<nsw> : i64
      %32 = llvm.mul %26, %17 overflow<nsw> : i64
      %33 = llvm.getelementptr %arg0[%31] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %34 = llvm.load %33 : !llvm.ptr -> i32
      %35 = llvm.add %34, %15 : i32
      %36 = llvm.mlir.constant(1 : i64) : i64
      %37 = llvm.mlir.constant(0 : i64) : i64
      %38 = llvm.sub %25, %36 overflow<nsw> : i64
      %39 = llvm.mul %38, %36 overflow<nsw> : i64
      %40 = llvm.mul %39, %36 overflow<nsw> : i64
      %41 = llvm.add %40, %37 overflow<nsw> : i64
      %42 = llvm.mul %36, %17 overflow<nsw> : i64
      %43 = llvm.getelementptr %arg1[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %35, %43 : i32, !llvm.ptr
      %44 = llvm.load %arg2 : !llvm.ptr -> i32
      %45 = llvm.add %44, %20 : i32
      %46 = llvm.sub %22, %16 : i64
      llvm.br ^bb1(%45, %46 : i32, i64)
    ^bb3:  // pred: ^bb1
      llvm.store %21, %arg2 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @_QFEa() {addr_space = 0 : i32} : !llvm.array<40 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<40 x i32>
    llvm.return %0 : !llvm.array<40 x i32>
  }
  llvm.mlir.global internal @_QFEb() {addr_space = 0 : i32} : !llvm.array<40 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<40 x i32>
    llvm.return %0 : !llvm.array<40 x i32>
  }
  llvm.mlir.global internal @_QFEc() {addr_space = 0 : i32} : !llvm.array<40 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<40 x i32>
    llvm.return %0 : !llvm.array<40 x i32>
  }
  llvm.mlir.global internal @_QFEn() {addr_space = 0 : i32} : i32 {
    %0 = llvm.mlir.constant(40 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.func @_FortranAProgramStart(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @_FortranAProgramEndStatement() attributes {sym_visibility = "private"}
  llvm.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.zero : !llvm.ptr
    llvm.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %1) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @_QQmain() : () -> ()
    llvm.call @_FortranAProgramEndStatement() : () -> ()
    llvm.return %0 : i32
  }
}

// %strucArg holds pointers to shared data.
// CHECK: define void @_QQmain() {
// CHECK-DAG: %[[STRUCTARG:.+]] = alloca { ptr, ptr, ptr }, align 8
// CHECK-DAG:  %[[DEP_ARRAY:.+]] = alloca [1 x %struct.kmp_dep_info], align 8
// CHECK: %[[DEP_INFO:.+]]  = getelementptr inbounds [1 x %struct.kmp_dep_info], ptr %[[DEP_ARRAY]], i64 0, i64 0
// CHECK: %[[PTR0:.+]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[DEP_INFO]], i32 0, i32 0
// CHECK: store i64 ptrtoint (ptr @_QFEa to i64), ptr %[[PTR0]], align 4
// CHECK: %[[PTR1:.+]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[DEP_INFO]], i32 0, i32 1
// CHECK: store i64 8, ptr %[[PTR1]], align 4
// CHECK: %[[PTR2:.+]] = getelementptr inbounds nuw %struct.kmp_dep_info, ptr %[[DEP_INFO]], i32 0, i32 2
// CHECK: store i8 1, ptr %[[PTR2]], align 1

// CHECK: %[[TASKDATA:.+]] = call ptr @__kmpc_omp_task_alloc({{.+}}, ptr @.omp_target_task_proxy_func)
// CHECK: %[[SHARED_DATA:.+]] = load ptr, ptr %[[TASKDATA]], align 8
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[SHARED_DATA]], ptr align 1 %[[STRUCTARG]], i64 24, i1 false)
// CHECK: call void @__kmpc_omp_wait_deps({{.+}}, i32 1, ptr %[[DEP_ARRAY]], i32 0, ptr null)
// CHECK: call void @__kmpc_omp_task_begin_if0({{.+}}, ptr  %[[TASKDATA]])
// CHECK: call void @.omp_target_task_proxy_func({{.+}}, ptr %[[TASKDATA]])
// CHECK: call void @__kmpc_omp_task_complete_if0({{.+}}, ptr %[[TASKDATA]])
	      
