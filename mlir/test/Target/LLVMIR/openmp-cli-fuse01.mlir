// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --enable-var-scope


llvm.func @fuse_trivial_loops(%baseptr: !llvm.ptr, %tc1: i32, %tc2: i32) -> () {
  %literal_cli1 = omp.new_cli
  omp.canonical_loop(%literal_cli1) %iv1 : i32 in range(%tc1) {
    %ptr = llvm.getelementptr inbounds %baseptr[%iv1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %val = llvm.mlir.constant(42.0 : f32) : f32
    llvm.store %val, %ptr : f32, !llvm.ptr
    omp.terminator
  }
  %literal_cli2 = omp.new_cli
  omp.canonical_loop(%literal_cli2) %iv2 : i32 in range(%tc2) {
    %ptr = llvm.getelementptr inbounds %baseptr[%iv2] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %val = llvm.mlir.constant(21.0 : f32) : f32
    llvm.store %val, %ptr : f32, !llvm.ptr
    omp.terminator
  }
  omp.fuse <- (%literal_cli1, %literal_cli2)
  llvm.return
}

// CHECK-LABEL:    define void @fuse_trivial_loops(
// CHECK-SAME:       ptr %[[VAL_11:.+]], i32 %[[VAL_5:.+]], i32 %[[VAL_16:.+]]) {
// CHECK-NEXT:       br label %[[OMP_OMP_LOOP_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_OMP_LOOP_PREHEADER]]:
// CHECK-NEXT:       br label %[[OMP_OMP_LOOP_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_OMP_LOOP_AFTER]]:
// CHECK-NEXT:       br label %[[OMP_OMP_LOOP_PREHEADER1:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_OMP_LOOP_PREHEADER1]]:
// CHECK-NEXT:       br label %[[OMP_FUSE_COMP_TC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSE_COMP_TC]]:
// CHECK-NEXT:       %[[VAL_15:.+]] = icmp sgt i32 %[[VAL_5:.+]], %[[VAL_16:.+]]
// CHECK-NEXT:       %[[VAL_17:.+]] = select i1 %[[VAL_15:.+]], i32 %[[VAL_5:.+]], i32 %[[VAL_16:.+]]
// CHECK-NEXT:       br label %[[OMP_FUSED_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_PREHEADER]]:
// CHECK-NEXT:       br label %[[OMP_FUSED_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_HEADER]]:
// CHECK-NEXT:       %[[VAL_4:.+]] = phi i32 [ 0, %[[VAL_18:.+]] ], [ %[[VAL_27:.+]], %[[VAL_26:.+]] ]
// CHECK-NEXT:       br label %[[OMP_FUSED_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_COND]]:
// CHECK-NEXT:       %[[VAL_29:.+]] = icmp ult i32 %[[VAL_4:.+]], %[[VAL_17:.+]]
// CHECK-NEXT:       br i1 %[[VAL_29:.+]], label %[[OMP_FUSED_BODY:.+]], label %[[OMP_FUSED_EXIT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_BODY]]:
// CHECK-NEXT:       br label %[[OMP_FUSED_INNER_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_INNER_COND]]:
// CHECK-NEXT:       %[[VAL_3:.+]] = icmp slt i32 %[[VAL_4:.+]], %[[VAL_5:.+]]
// CHECK-NEXT:       br i1 %[[VAL_3:.+]], label %[[OMP_OMP_LOOP_BODY:.+]], label %[[OMP_FUSED_INNER_COND13:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_OMP_LOOP_BODY]]:
// CHECK-NEXT:       br label %[[OMP_LOOP_REGION:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_LOOP_REGION]]:
// CHECK-NEXT:       %[[VAL_10:.+]] = getelementptr inbounds float, ptr %[[VAL_11:.+]], i32 %[[VAL_4:.+]]
// CHECK-NEXT:       store float 4.200000e+01, ptr %[[VAL_10:.+]], align 4
// CHECK-NEXT:       br label %[[OMP_REGION_CONT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_REGION_CONT]]:
// CHECK-NEXT:       br label %[[OMP_FUSED_INNER_COND13:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_INNER_COND13]]:
// CHECK-NEXT:       %[[VAL_19:.+]] = icmp slt i32 %[[VAL_4:.+]], %[[VAL_16:.+]]
// CHECK-NEXT:       br i1 %[[VAL_19:.+]], label %[[OMP_OMP_LOOP_BODY4:.+]], label %[[OMP_FUSED_PRE_LATCH:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_OMP_LOOP_BODY4]]:
// CHECK-NEXT:       br label %[[OMP_LOOP_REGION12:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_LOOP_REGION12]]:
// CHECK-NEXT:       %[[VAL_23:.+]] = getelementptr inbounds float, ptr %[[VAL_11:.+]], i32 %[[VAL_4:.+]]
// CHECK-NEXT:       store float 2.100000e+01, ptr %[[VAL_23:.+]], align 4
// CHECK-NEXT:       br label %[[OMP_REGION_CONT11:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_REGION_CONT11]]:
// CHECK-NEXT:       br label %[[OMP_FUSED_PRE_LATCH:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_PRE_LATCH]]:
// CHECK-NEXT:       br label %[[OMP_FUSED_INC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_INC]]:
// CHECK-NEXT:       %[[VAL_27:.+]] = add nuw i32 %[[VAL_4:.+]], 1
// CHECK-NEXT:       br label %[[OMP_FUSED_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_EXIT]]:
// CHECK-NEXT:       br label %[[OMP_FUSED_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_FUSED_AFTER]]:
// CHECK-NEXT:       br label %[[OMP_OMP_LOOP_AFTER7:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:     [[OMP_OMP_LOOP_AFTER7]]:
// CHECK-NEXT:       ret void

