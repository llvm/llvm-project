// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --enable-var-scope


llvm.func @fuse_looprange_loops(%baseptr: !llvm.ptr, %tc1: i32, %tc2: i32, %tc3: i32) -> () {
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
  %literal_cli3 = omp.new_cli
  omp.canonical_loop(%literal_cli3) %iv3 : i32 in range(%tc3) {
    %ptr = llvm.getelementptr inbounds %baseptr[%iv3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %val = llvm.mlir.constant(63.0 : f32) : f32
    llvm.store %val, %ptr : f32, !llvm.ptr
    omp.terminator
  }
  omp.fuse <- (%literal_cli1, %literal_cli2, %literal_cli3) looprange(first = 1, count = 2)
  llvm.return
}


// CHECK-LABEL:   define void @fuse_looprange_loops(
// CHECK-SAME:      ptr %[[VAL_23:.+]], i32 %[[VAL_5:.+]], i32 %[[VAL_6:.+]], i32 %[[VAL_40:.+]]) {
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_PREHEADER]]:
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_AFTER]]:
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_PREHEADER1:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_PREHEADER1]]:
// CHECK-NEXT:      br label %[[OMP_FUSE_COMP_TC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSE_COMP_TC]]:
// CHECK-NEXT:      %[[VAL_4:.+]] = icmp sgt i32 %[[VAL_5:.+]], %[[VAL_6:.+]]
// CHECK-NEXT:      %[[VAL_7:.+]] = select i1 %[[VAL_4:.+]], i32 %[[VAL_5:.+]], i32 %[[VAL_6:.+]]
// CHECK-NEXT:      br label %[[OMP_FUSED_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_PREHEADER]]:
// CHECK-NEXT:      br label %[[OMP_FUSED_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_HEADER]]:
// CHECK-NEXT:      %[[VAL_11:.+]] = phi i32 [ 0, %[[VAL_8:.+]] ], [ %[[VAL_12:.+]], %[[VAL_10:.+]] ]
// CHECK-NEXT:      br label %[[OMP_FUSED_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_COND]]:
// CHECK-NEXT:      %[[VAL_14:.+]] = icmp ult i32 %[[VAL_11:.+]], %[[VAL_7:.+]]
// CHECK-NEXT:      br i1 %[[VAL_14:.+]], label %[[OMP_FUSED_BODY:.+]], label %[[OMP_FUSED_EXIT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_BODY]]:
// CHECK-NEXT:      br label %[[OMP_FUSED_INNER_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_INNER_COND]]:
// CHECK-NEXT:      %[[VAL_18:.+]] = icmp slt i32 %[[VAL_11:.+]], %[[VAL_5:.+]]
// CHECK-NEXT:      br i1 %[[VAL_18:.+]], label %[[OMP_OMP_LOOP_BODY:.+]], label %[[OMP_FUSED_INNER_COND25:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_BODY]]:
// CHECK-NEXT:      br label %[[OMP_LOOP_REGION:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_LOOP_REGION]]:
// CHECK-NEXT:      %[[VAL_22:.+]] = getelementptr inbounds float, ptr %[[VAL_23:.+]], i32 %[[VAL_11:.+]]
// CHECK-NEXT:      store float 4.200000e+01, ptr %[[VAL_22:.+]], align 4
// CHECK-NEXT:      br label %[[OMP_REGION_CONT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_REGION_CONT]]:
// CHECK-NEXT:      br label %[[OMP_FUSED_INNER_COND25:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_INNER_COND25]]:
// CHECK-NEXT:      %[[VAL_25:.+]] = icmp slt i32 %[[VAL_11:.+]], %[[VAL_6:.+]]
// CHECK-NEXT:      br i1 %[[VAL_25:.+]], label %[[OMP_OMP_LOOP_BODY4:.+]], label %[[OMP_FUSED_PRE_LATCH:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_BODY4]]:
// CHECK-NEXT:      br label %[[OMP_LOOP_REGION12:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_LOOP_REGION12]]:
// CHECK-NEXT:      %[[VAL_29:.+]] = getelementptr inbounds float, ptr %[[VAL_23:.+]], i32 %[[VAL_11:.+]]
// CHECK-NEXT:      store float 2.100000e+01, ptr %[[VAL_29:.+]], align 4
// CHECK-NEXT:      br label %[[OMP_REGION_CONT11:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_REGION_CONT11]]:
// CHECK-NEXT:      br label %[[OMP_FUSED_PRE_LATCH:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_PRE_LATCH]]:
// CHECK-NEXT:      br label %[[OMP_FUSED_INC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_INC]]:
// CHECK-NEXT:      %[[VAL_12:.+]] = add nuw i32 %[[VAL_11:.+]], 1
// CHECK-NEXT:      br label %[[OMP_FUSED_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_EXIT]]:
// CHECK-NEXT:      br label %[[OMP_FUSED_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_FUSED_AFTER]]:
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_AFTER7:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_AFTER7]]:
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_PREHEADER13:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_PREHEADER13]]:
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_HEADER14:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_HEADER14]]:
// CHECK-NEXT:      %[[VAL_36:.+]] = phi i32 [ 0, %[[VAL_33:.+]] ], [ %[[VAL_37:.+]], %[[VAL_35:.+]] ]
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_COND15:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_COND15]]:
// CHECK-NEXT:      %[[VAL_39:.+]] = icmp ult i32 %[[VAL_36:.+]], %[[VAL_40:.+]]
// CHECK-NEXT:      br i1 %[[VAL_39:.+]], label %[[OMP_OMP_LOOP_BODY16:.+]], label %[[OMP_OMP_LOOP_EXIT18:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_BODY16]]:
// CHECK-NEXT:      br label %[[OMP_LOOP_REGION24:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_LOOP_REGION24]]:
// CHECK-NEXT:      %[[VAL_44:.+]] = getelementptr inbounds float, ptr %[[VAL_23:.+]], i32 %[[VAL_36:.+]]
// CHECK-NEXT:      store float 6.300000e+01, ptr %[[VAL_44:.+]], align 4
// CHECK-NEXT:      br label %[[OMP_REGION_CONT23:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_REGION_CONT23]]:
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_INC17:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_INC17]]:
// CHECK-NEXT:      %[[VAL_37:.+]] = add nuw i32 %[[VAL_36:.+]], 1
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_HEADER14:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_EXIT18]]:
// CHECK-NEXT:      br label %[[OMP_OMP_LOOP_AFTER19:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:    [[OMP_OMP_LOOP_AFTER19]]:
// CHECK-NEXT:      ret void

