// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --enable-var-scope


llvm.func @tile_trivial_loop(%baseptr: !llvm.ptr, %tc: i32, %ts: i32) -> () {
  %literal_cli = omp.new_cli
  omp.canonical_loop(%literal_cli) %iv : i32 in range(%tc) {
    %ptr = llvm.getelementptr inbounds %baseptr[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %val = llvm.mlir.constant(42.0 : f32) : f32
    llvm.store %val, %ptr : f32, !llvm.ptr
    omp.terminator
  }
  omp.tile <- (%literal_cli) sizes(%ts : i32)
  llvm.return
}


// CHECK-LABEL: define void @tile_trivial_loop(
// CHECK-SAME:    ptr %[[TMP0:.+]], i32 %[[TMP1:.+]], i32 %[[TMP2:.+]]) {
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_PREHEADER]]:
// CHECK-NEXT:    %[[TMP4:.+]] = udiv i32 %[[TMP1:.+]], %[[TMP2:.+]]
// CHECK-NEXT:    %[[TMP5:.+]] = urem i32 %[[TMP1:.+]], %[[TMP2:.+]]
// CHECK-NEXT:    %[[TMP6:.+]] = icmp ne i32 %[[TMP5:.+]], 0
// CHECK-NEXT:    %[[TMP7:.+]] = zext i1 %[[TMP6:.+]] to i32
// CHECK-NEXT:    %[[OMP_FLOOR0_TRIPCOUNT:.+]] = add nuw i32 %[[TMP4:.+]], %[[TMP7:.+]]
// CHECK-NEXT:    br label %[[OMP_FLOOR0_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_PREHEADER]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_HEADER]]:
// CHECK-NEXT:    %[[OMP_FLOOR0_IV:.+]] = phi i32 [ 0, %[[OMP_FLOOR0_PREHEADER:.+]] ], [ %[[OMP_FLOOR0_NEXT:.+]], %[[OMP_FLOOR0_INC:.+]] ]
// CHECK-NEXT:    br label %[[OMP_FLOOR0_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_COND]]:
// CHECK-NEXT:    %[[OMP_FLOOR0_CMP:.+]] = icmp ult i32 %[[OMP_FLOOR0_IV:.+]], %[[OMP_FLOOR0_TRIPCOUNT:.+]]
// CHECK-NEXT:    br i1 %[[OMP_FLOOR0_CMP:.+]], label %[[OMP_FLOOR0_BODY:.+]], label %[[OMP_FLOOR0_EXIT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_BODY]]:
// CHECK-NEXT:    %[[TMP8:.+]] = icmp eq i32 %[[OMP_FLOOR0_IV:.+]], %[[TMP4:.+]]
// CHECK-NEXT:    %[[TMP9:.+]] = select i1 %[[TMP8:.+]], i32 %[[TMP5:.+]], i32 %[[TMP2:.+]]
// CHECK-NEXT:    br label %[[OMP_TILE0_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_PREHEADER]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_HEADER]]:
// CHECK-NEXT:    %[[OMP_TILE0_IV:.+]] = phi i32 [ 0, %[[OMP_TILE0_PREHEADER:.+]] ], [ %[[OMP_TILE0_NEXT:.+]], %[[OMP_TILE0_INC:.+]] ]
// CHECK-NEXT:    br label %[[OMP_TILE0_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_COND]]:
// CHECK-NEXT:    %[[OMP_TILE0_CMP:.+]] = icmp ult i32 %[[OMP_TILE0_IV:.+]], %[[TMP9:.+]]
// CHECK-NEXT:    br i1 %[[OMP_TILE0_CMP:.+]], label %[[OMP_TILE0_BODY:.+]], label %[[OMP_TILE0_EXIT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_BODY]]:
// CHECK-NEXT:    %[[TMP10:.+]] = mul nuw i32 %[[TMP2:.+]], %[[OMP_FLOOR0_IV:.+]]
// CHECK-NEXT:    %[[TMP11:.+]] = add nuw i32 %[[TMP10:.+]], %[[OMP_TILE0_IV:.+]]
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_BODY:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_BODY]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_REGION:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_LOOP_REGION]]:
// CHECK-NEXT:    %[[TMP12:.+]] = getelementptr inbounds float, ptr %[[TMP0:.+]], i32 %[[TMP11:.+]]
// CHECK-NEXT:    store float 4.200000e+01, ptr %[[TMP12:.+]], align 4
// CHECK-NEXT:    br label %[[OMP_REGION_CONT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_REGION_CONT]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_INC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_INC]]:
// CHECK-NEXT:    %[[OMP_TILE0_NEXT:.+]] = add nuw i32 %[[OMP_TILE0_IV:.+]], 1
// CHECK-NEXT:    br label %[[OMP_TILE0_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_AFTER]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_INC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_INC]]:
// CHECK-NEXT:    %[[OMP_FLOOR0_NEXT:.+]] = add nuw i32 %[[OMP_FLOOR0_IV:.+]], 1
// CHECK-NEXT:    br label %[[OMP_FLOOR0_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_AFTER]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_AFTER]]:
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }
