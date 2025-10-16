// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --enable-var-scope


llvm.func @tile_2d_loop(%baseptr: !llvm.ptr, %tc1: i32, %tc2: i32, %ts1: i32, %ts2: i32) -> () {
  %literal_outer = omp.new_cli
  %literal_inner = omp.new_cli
  omp.canonical_loop(%literal_outer) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%literal_inner) %iv2 : i32 in range(%tc2) {
      %idx = llvm.add %iv1, %iv2 : i32
      %ptr = llvm.getelementptr inbounds %baseptr[%idx] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %val = llvm.mlir.constant(42.0 : f32) : f32
      llvm.store %val, %ptr : f32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  omp.tile <- (%literal_outer, %literal_inner) sizes(%ts1, %ts2 : i32,i32)
  llvm.return
}


// CHECK-LABEL: define void @tile_2d_loop(
// CHECK-SAME:      ptr %[[TMP0:.+]], i32 %[[TMP1:.+]], i32 %[[TMP2:.+]], i32 %[[TMP3:.+]], i32 %[[TMP4:.+]]) {
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_PREHEADER]]:
// CHECK-NEXT:    %[[TMP6:.+]] = udiv i32 %[[TMP1:.+]], %[[TMP3:.+]]
// CHECK-NEXT:    %[[TMP7:.+]] = urem i32 %[[TMP1:.+]], %[[TMP3:.+]]
// CHECK-NEXT:    %[[TMP8:.+]] = icmp ne i32 %[[TMP7:.+]], 0
// CHECK-NEXT:    %[[TMP9:.+]] = zext i1 %[[TMP8:.+]] to i32
// CHECK-NEXT:    %[[OMP_FLOOR0_TRIPCOUNT:.+]] = add nuw i32 %[[TMP6:.+]], %[[TMP9:.+]]
// CHECK-NEXT:    %[[TMP10:.+]] = udiv i32 %[[TMP2:.+]], %[[TMP4:.+]]
// CHECK-NEXT:    %[[TMP11:.+]] = urem i32 %[[TMP2:.+]], %[[TMP4:.+]]
// CHECK-NEXT:    %[[TMP12:.+]] = icmp ne i32 %[[TMP11:.+]], 0
// CHECK-NEXT:    %[[TMP13:.+]] = zext i1 %[[TMP12:.+]] to i32
// CHECK-NEXT:    %[[OMP_FLOOR1_TRIPCOUNT:.+]] = add nuw i32 %[[TMP10:.+]], %[[TMP13:.+]]
// CHECK-NEXT:    br label %[[OMP_FLOOR0_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_HEADER:.+]]:
// CHECK-NEXT:    %[[OMP_OMP_LOOP_IV:.+]] = phi i32 [ %[[OMP_OMP_LOOP_NEXT:.+]], %[[OMP_OMP_LOOP_INC:.+]] ]
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_COND]]:
// CHECK-NEXT:    %[[OMP_OMP_LOOP_CMP:.+]] = icmp ult i32 %[[TMP19:.+]], %[[TMP1:.+]]
// CHECK-NEXT:    br i1 %[[OMP_OMP_LOOP_CMP:.+]], label %[[OMP_OMP_LOOP_BODY:.+]], label %[[OMP_OMP_LOOP_EXIT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_BODY]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_REGION:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_LOOP_REGION]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_PREHEADER1:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_PREHEADER1]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_BODY4:.+]]
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
// CHECK-NEXT:    br label %[[OMP_FLOOR1_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR1_PREHEADER]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR1_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR1_HEADER]]:
// CHECK-NEXT:    %[[OMP_FLOOR1_IV:.+]] = phi i32 [ 0, %[[OMP_FLOOR1_PREHEADER:.+]] ], [ %[[OMP_FLOOR1_NEXT:.+]], %[[OMP_FLOOR1_INC:.+]] ]
// CHECK-NEXT:    br label %[[OMP_FLOOR1_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR1_COND]]:
// CHECK-NEXT:    %[[OMP_FLOOR1_CMP:.+]] = icmp ult i32 %[[OMP_FLOOR1_IV:.+]], %[[OMP_FLOOR1_TRIPCOUNT:.+]]
// CHECK-NEXT:    br i1 %[[OMP_FLOOR1_CMP:.+]], label %[[OMP_FLOOR1_BODY:.+]], label %[[OMP_FLOOR1_EXIT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR1_BODY]]:
// CHECK-NEXT:    %[[TMP14:.+]] = icmp eq i32 %[[OMP_FLOOR0_IV:.+]], %[[TMP6:.+]]
// CHECK-NEXT:    %[[TMP15:.+]] = select i1 %[[TMP14:.+]], i32 %[[TMP7:.+]], i32 %[[TMP3:.+]]
// CHECK-NEXT:    %[[TMP16:.+]] = icmp eq i32 %[[OMP_FLOOR1_IV:.+]], %[[TMP10:.+]]
// CHECK-NEXT:    %[[TMP17:.+]] = select i1 %[[TMP16:.+]], i32 %[[TMP11:.+]], i32 %[[TMP4:.+]]
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
// CHECK-NEXT:    %[[OMP_TILE0_CMP:.+]] = icmp ult i32 %[[OMP_TILE0_IV:.+]], %[[TMP15:.+]]
// CHECK-NEXT:    br i1 %[[OMP_TILE0_CMP:.+]], label %[[OMP_TILE0_BODY:.+]], label %[[OMP_TILE0_EXIT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_BODY]]:
// CHECK-NEXT:    br label %[[OMP_TILE1_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE1_PREHEADER]]:
// CHECK-NEXT:    br label %[[OMP_TILE1_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE1_HEADER]]:
// CHECK-NEXT:    %[[OMP_TILE1_IV:.+]] = phi i32 [ 0, %[[OMP_TILE1_PREHEADER:.+]] ], [ %[[OMP_TILE1_NEXT:.+]], %[[OMP_TILE1_INC:.+]] ]
// CHECK-NEXT:    br label %[[OMP_TILE1_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE1_COND]]:
// CHECK-NEXT:    %[[OMP_TILE1_CMP:.+]] = icmp ult i32 %[[OMP_TILE1_IV:.+]], %[[TMP17:.+]]
// CHECK-NEXT:    br i1 %[[OMP_TILE1_CMP:.+]], label %[[OMP_TILE1_BODY:.+]], label %[[OMP_TILE1_EXIT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE1_BODY]]:
// CHECK-NEXT:    %[[TMP18:.+]] = mul nuw i32 %[[TMP3:.+]], %[[OMP_FLOOR0_IV:.+]]
// CHECK-NEXT:    %[[TMP19:.+]] = add nuw i32 %[[TMP18:.+]], %[[OMP_TILE0_IV:.+]]
// CHECK-NEXT:    %[[TMP20:.+]] = mul nuw i32 %[[TMP4:.+]], %[[OMP_FLOOR1_IV:.+]]
// CHECK-NEXT:    %[[TMP21:.+]] = add nuw i32 %[[TMP20:.+]], %[[OMP_TILE1_IV:.+]]
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_BODY:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_BODY4]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_REGION12:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_LOOP_REGION12]]:
// CHECK-NEXT:    %[[TMP22:.+]] = add i32 %[[TMP19:.+]], %[[TMP21:.+]]
// CHECK-NEXT:    %[[TMP23:.+]] = getelementptr inbounds float, ptr %[[TMP0:.+]], i32 %[[TMP22:.+]]
// CHECK-NEXT:    store float 4.200000e+01, ptr %[[TMP23:.+]], align 4
// CHECK-NEXT:    br label %[[OMP_REGION_CONT11:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_REGION_CONT11]]:
// CHECK-NEXT:    br label %[[OMP_TILE1_INC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE1_INC]]:
// CHECK-NEXT:    %[[OMP_TILE1_NEXT:.+]] = add nuw i32 %[[OMP_TILE1_IV:.+]], 1
// CHECK-NEXT:    br label %[[OMP_TILE1_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE1_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_TILE1_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE1_AFTER]]:
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
// CHECK-NEXT:    br label %[[OMP_FLOOR1_INC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR1_INC]]:
// CHECK-NEXT:    %[[OMP_FLOOR1_NEXT:.+]] = add nuw i32 %[[OMP_FLOOR1_IV:.+]], 1
// CHECK-NEXT:    br label %[[OMP_FLOOR1_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR1_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR1_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR1_AFTER]]:
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
// CHECK-NEXT:  [[OMP_REGION_CONT:.+]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_INC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_INC]]:
// CHECK-NEXT:    %[[OMP_OMP_LOOP_NEXT:.+]] = add nuw i32 %[[TMP19:.+]], 1
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_AFTER]]:
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }
