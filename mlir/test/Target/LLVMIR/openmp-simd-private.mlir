// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

omp.private {type = private} @i_privatizer : i32

// CHECK-LABEL: test_loop_var_privatization()
//                Original (non-privatized) allocation for `i`.
// CHECK:         %{{.*}} = alloca i32, i64 1, align 4
// CHECK:         %[[DUMMY:.*]] = alloca float, i64 1, align 4
// CHECK:         %[[PRIV_I:.*]] = alloca i32, align 4
// CHECK:         br label %[[LATE_ALLOC:.*]]

// CHECK:     [[LATE_ALLOC]]:
// CHECK:         br label %[[AFTER_ALLOC:.*]]

// CHECK:       [[AFTER_ALLOC]]:
// CHECK:         br label %[[ENTRY:.*]]

// CHECK:       [[ENTRY]]:
// CHECK:         br label %[[OMP_LOOP_PREHEADER:.*]]

// CHECK:       [[OMP_LOOP_PREHEADER]]:
// CHECK:         br label %[[OMP_LOOP_HEADER:.*]]

// CHECK:       [[OMP_LOOP_HEADER]]:
// CHECK:         %[[OMP_LOOP_IV:.*]] = phi i32 [ 0, %[[OMP_LOOP_PREHEADER]] ], [ %[[OMP_LOOP_NEXT:.*]], %[[OMP_LOOP_INC:.*]] ]
// CHECK:         br label %[[OMP_LOOP_COND:.*]]

// CHECK:       [[OMP_LOOP_COND]]:
// CHECK:         %[[OMP_LOOP_CMP:.*]] = icmp ult i32 %[[OMP_LOOP_IV]], 10
// CHECK:         br i1 %[[OMP_LOOP_CMP]], label %[[OMP_LOOP_BODY:.*]], label %[[OMP_LOOP_EXIT:.*]]

// CHECK:       [[OMP_LOOP_BODY]]:
// CHECK:         %[[IV_UPDATE:.*]] = mul i32 %[[OMP_LOOP_IV]], 1
// CHECK:         %[[IV_UPDATE_2:.*]] = add i32 %[[IV_UPDATE]], 1
// CHECK:         br label %[[OMP_SIMD_REGION:.*]]

// CHECK:       [[OMP_SIMD_REGION]]:
// CHECK:         store i32 %[[IV_UPDATE_2]], ptr %[[PRIV_I]], align 4
// CHECK:         %[[DUMMY_VAL:.*]] = load float, ptr %[[DUMMY]], align 4
// CHECK:         %[[PRIV_I_VAL:.*]] = load i32, ptr %[[PRIV_I]], align 4
// CHECK:         %[[PRIV_I_VAL_FLT:.*]] = sitofp i32 %[[PRIV_I_VAL]] to float
// CHECK:         %[[DUMMY_VAL_UPDATE:.*]] = fadd {{.*}} float %[[DUMMY_VAL]], %[[PRIV_I_VAL_FLT]]
// CHECK:         store float %[[DUMMY_VAL_UPDATE]], ptr %[[DUMMY]], align 4, !llvm.access.group !1
// CHECK:        br label %[[OMP_REGION_CONT:.*]]

// CHECK:      [[OMP_REGION_CONT]]:
// CHECK:        br label %[[OMP_LOOP_INC:.*]]

// CHECK:      [[OMP_LOOP_INC]]:
// CHECK:        %[[OMP_LOOP_NEXT:.*]] = add nuw i32 %[[OMP_LOOP_IV]], 1
// CHECK:        br label %[[OMP_LOOP_HEADER]]

// CHECK:      [[OMP_LOOP_EXIT]]:


llvm.func @test_loop_var_privatization() attributes {fir.internal_name = "_QPtest_private_clause"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x f32 {bindc_name = "dummy"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(10 : i32) : i32
  %5 = llvm.mlir.constant(1 : i32) : i32
  omp.simd private(@i_privatizer %1 -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%arg1) : i32 = (%5) to (%4) inclusive step (%5) {
      llvm.store %arg1, %arg0 : i32, !llvm.ptr
      %8 = llvm.load %3 : !llvm.ptr -> f32
      %9 = llvm.load %arg0 : !llvm.ptr -> i32
      %10 = llvm.sitofp %9 : i32 to f32
      %11 = llvm.fadd %8, %10 {fastmathFlags = #llvm.fastmath<contract>} : f32
      llvm.store %11, %3 : f32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

omp.private {type = private} @dummy_privatizer : f32

// CHECK-LABEL: test_private_clause()
//                Original (non-privatized) allocation for `i`.
// CHECK:         %{{.*}} = alloca i32, i64 1, align 4
//                Original (non-privatized) allocation for `dummy`.
// CHECK:         %{{.*}} = alloca float, i64 1, align 4
// CHECK:         %[[PRIV_DUMMY:.*]] = alloca float, align 4
// CHECK:         %[[PRIV_I:.*]] = alloca i32, align 4

// CHECK:       omp.simd.region:
// CHECK-NOT:     br label
// CHECK:         store i32 %{{.*}}, ptr %[[PRIV_I]], align 4
// CHECK:        %{{.*}} = load float, ptr %[[PRIV_DUMMY]], align 4
// CHECK:        store float %{{.*}}, ptr %[[PRIV_DUMMY]], align 4

llvm.func @test_private_clause() attributes {fir.internal_name = "_QPtest_private_clause"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x f32 {bindc_name = "dummy"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(10 : i32) : i32
  %5 = llvm.mlir.constant(1 : i32) : i32
  omp.simd private(@dummy_privatizer %3 -> %arg0, @i_privatizer %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2) : i32 = (%5) to (%4) inclusive step (%5) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %8 = llvm.load %arg0 : !llvm.ptr -> f32
      %9 = llvm.load %arg1 : !llvm.ptr -> i32
      %10 = llvm.sitofp %9 : i32 to f32
      %11 = llvm.fadd %8, %10 {fastmathFlags = #llvm.fastmath<contract>} : f32
      llvm.store %11, %arg0 : f32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}
