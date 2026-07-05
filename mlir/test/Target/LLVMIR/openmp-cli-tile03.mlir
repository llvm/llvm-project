// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --enable-var-scope


llvm.func @tile_composition(%baseptr: !llvm.ptr, %tc: i32, %ts: i32, %grid_ts: i32, %intratile_ts: i32) -> () {
  %canonloop = omp.new_cli
  %grid = omp.new_cli
  %intratile = omp.new_cli
  %grid_intratile = omp.new_cli
  %grid_grid = omp.new_cli
  %intratile_grid = omp.new_cli
  %intratile_intratile = omp.new_cli

  omp.canonical_loop(%canonloop) %idx : i32 in range(%tc) {
    %ptr = llvm.getelementptr inbounds %baseptr[%idx] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %val = llvm.mlir.constant(42.0 : f32) : f32
    llvm.store %val, %ptr : f32, !llvm.ptr
    omp.terminator
  }

  omp.tile(%grid, %intratile) <- (%canonloop) sizes(%ts : i32)
  omp.tile(%grid_grid, %grid_intratile) <- (%grid) sizes(%grid_ts : i32)
  omp.tile(%intratile_grid, %intratile_intratile) <- (%intratile) sizes(%intratile_ts : i32)
  llvm.return
}


// CHECK-LABEL: define void @tile_composition(
// CHECK-SAME:      ptr %[[TMP0:.+]], i32 %[[TMP1:.+]], i32 %[[TMP2:.+]], i32 %[[TMP3:.+]], i32 %[[TMP4:.+]]) {
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_PREHEADER]]:
// CHECK-NEXT:    %[[TMP6:.+]] = udiv i32 %[[TMP1:.+]], %[[TMP2:.+]]
// CHECK-NEXT:    %[[TMP7:.+]] = urem i32 %[[TMP1:.+]], %[[TMP2:.+]]
// CHECK-NEXT:    %[[TMP8:.+]] = icmp ne i32 %[[TMP7:.+]], 0
// CHECK-NEXT:    %[[TMP9:.+]] = zext i1 %[[TMP8:.+]] to i32
// CHECK-NEXT:    %[[OMP_FLOOR0_TRIPCOUNT:.+]] = add nuw i32 %[[TMP6:.+]], %[[TMP9:.+]]
// CHECK-NEXT:    br label %[[OMP_FLOOR0_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_PREHEADER]]:
// CHECK-NEXT:    %[[TMP10:.+]] = udiv i32 %[[OMP_FLOOR0_TRIPCOUNT:.+]], %[[TMP3:.+]]
// CHECK-NEXT:    %[[TMP11:.+]] = urem i32 %[[OMP_FLOOR0_TRIPCOUNT:.+]], %[[TMP3:.+]]
// CHECK-NEXT:    %[[TMP12:.+]] = icmp ne i32 %[[TMP11:.+]], 0
// CHECK-NEXT:    %[[TMP13:.+]] = zext i1 %[[TMP12:.+]] to i32
// CHECK-NEXT:    %[[OMP_FLOOR0_TRIPCOUNT1:.+]] = add nuw i32 %[[TMP10:.+]], %[[TMP13:.+]]
// CHECK-NEXT:    br label %[[OMP_FLOOR0_PREHEADER2:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_PREHEADER2]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_HEADER3:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_HEADER3]]:
// CHECK-NEXT:    %[[OMP_FLOOR0_IV9:.+]] = phi i32 [ 0, %[[OMP_FLOOR0_PREHEADER2:.+]] ], [ %[[OMP_FLOOR0_NEXT11:.+]], %[[OMP_FLOOR0_INC6:.+]] ]
// CHECK-NEXT:    br label %[[OMP_FLOOR0_COND4:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_COND4]]:
// CHECK-NEXT:    %[[OMP_FLOOR0_CMP10:.+]] = icmp ult i32 %[[OMP_FLOOR0_IV9:.+]], %[[OMP_FLOOR0_TRIPCOUNT1:.+]]
// CHECK-NEXT:    br i1 %[[OMP_FLOOR0_CMP10:.+]], label %[[OMP_FLOOR0_BODY5:.+]], label %[[OMP_FLOOR0_EXIT7:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_BODY5]]:
// CHECK-NEXT:    %[[TMP14:.+]] = icmp eq i32 %[[OMP_FLOOR0_IV9:.+]], %[[TMP10:.+]]
// CHECK-NEXT:    %[[TMP15:.+]] = select i1 %[[TMP14:.+]], i32 %[[TMP11:.+]], i32 %[[TMP3:.+]]
// CHECK-NEXT:    br label %[[OMP_TILE0_PREHEADER12:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_PREHEADER12]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_HEADER13:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_HEADER13]]:
// CHECK-NEXT:    %[[OMP_TILE0_IV19:.+]] = phi i32 [ 0, %[[OMP_TILE0_PREHEADER12:.+]] ], [ %[[OMP_TILE0_NEXT21:.+]], %[[OMP_TILE0_INC16:.+]] ]
// CHECK-NEXT:    br label %[[OMP_TILE0_COND14:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_COND14]]:
// CHECK-NEXT:    %[[OMP_TILE0_CMP20:.+]] = icmp ult i32 %[[OMP_TILE0_IV19:.+]], %[[TMP15:.+]]
// CHECK-NEXT:    br i1 %[[OMP_TILE0_CMP20:.+]], label %[[OMP_TILE0_BODY15:.+]], label %[[OMP_TILE0_EXIT17:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_BODY15]]:
// CHECK-NEXT:    %[[TMP16:.+]] = mul nuw i32 %[[TMP3:.+]], %[[OMP_FLOOR0_IV9:.+]]
// CHECK-NEXT:    %[[TMP17:.+]] = add nuw i32 %[[TMP16:.+]], %[[OMP_TILE0_IV19:.+]]
// CHECK-NEXT:    br label %[[OMP_FLOOR0_BODY:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_BODY]]:
// CHECK-NEXT:    %[[TMP18:.+]] = icmp eq i32 %[[TMP17:.+]], %[[TMP6:.+]]
// CHECK-NEXT:    %[[TMP19:.+]] = select i1 %[[TMP18:.+]], i32 %[[TMP7:.+]], i32 %[[TMP2:.+]]
// CHECK-NEXT:    br label %[[OMP_TILE0_PREHEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_PREHEADER]]:
// CHECK-NEXT:    %[[TMP20:.+]] = udiv i32 %[[TMP19:.+]], %[[TMP4:.+]]
// CHECK-NEXT:    %[[TMP21:.+]] = urem i32 %[[TMP19:.+]], %[[TMP4:.+]]
// CHECK-NEXT:    %[[TMP22:.+]] = icmp ne i32 %[[TMP21:.+]], 0
// CHECK-NEXT:    %[[TMP23:.+]] = zext i1 %[[TMP22:.+]] to i32
// CHECK-NEXT:    %[[OMP_FLOOR0_TRIPCOUNT22:.+]] = add nuw i32 %[[TMP20:.+]], %[[TMP23:.+]]
// CHECK-NEXT:    br label %[[OMP_FLOOR0_PREHEADER23:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_PREHEADER23]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_HEADER]]:
// CHECK-NEXT:    %[[OMP_FLOOR0_IV:.+]] = phi i32 [ 0, %[[OMP_FLOOR0_PREHEADER23:.+]] ], [ %[[OMP_FLOOR0_NEXT:.+]], %[[OMP_FLOOR0_INC:.+]] ]
// CHECK-NEXT:    br label %[[OMP_FLOOR0_COND:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_COND]]:
// CHECK-NEXT:    %[[OMP_FLOOR0_CMP:.+]] = icmp ult i32 %[[OMP_FLOOR0_IV:.+]], %[[OMP_FLOOR0_TRIPCOUNT22:.+]]
// CHECK-NEXT:    br i1 %[[OMP_FLOOR0_CMP:.+]], label %[[OMP_FLOOR0_BODY24:.+]], label %[[OMP_FLOOR0_EXIT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_BODY24]]:
// CHECK-NEXT:    %[[TMP24:.+]] = icmp eq i32 %[[OMP_FLOOR0_IV:.+]], %[[TMP20:.+]]
// CHECK-NEXT:    %[[TMP25:.+]] = select i1 %[[TMP24:.+]], i32 %[[TMP21:.+]], i32 %[[TMP4:.+]]
// CHECK-NEXT:    br label %[[OMP_TILE0_PREHEADER26:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_PREHEADER26]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_HEADER27:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_HEADER27]]:
// CHECK-NEXT:    %[[OMP_TILE0_IV33:.+]] = phi i32 [ 0, %[[OMP_TILE0_PREHEADER26:.+]] ], [ %[[OMP_TILE0_NEXT35:.+]], %[[OMP_TILE0_INC30:.+]] ]
// CHECK-NEXT:    br label %[[OMP_TILE0_COND28:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_COND28]]:
// CHECK-NEXT:    %[[OMP_TILE0_CMP34:.+]] = icmp ult i32 %[[OMP_TILE0_IV33:.+]], %[[TMP25:.+]]
// CHECK-NEXT:    br i1 %[[OMP_TILE0_CMP34:.+]], label %[[OMP_TILE0_BODY29:.+]], label %[[OMP_TILE0_EXIT31:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_BODY29]]:
// CHECK-NEXT:    %[[TMP26:.+]] = mul nuw i32 %[[TMP4:.+]], %[[OMP_FLOOR0_IV:.+]]
// CHECK-NEXT:    %[[TMP27:.+]] = add nuw i32 %[[TMP26:.+]], %[[OMP_TILE0_IV33:.+]]
// CHECK-NEXT:    br label %[[OMP_TILE0_BODY:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_BODY]]:
// CHECK-NEXT:    %[[TMP28:.+]] = mul nuw i32 %[[TMP2:.+]], %[[TMP17:.+]]
// CHECK-NEXT:    %[[TMP29:.+]] = add nuw i32 %[[TMP28:.+]], %[[TMP27:.+]]
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_BODY:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_BODY]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_REGION:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_LOOP_REGION]]:
// CHECK-NEXT:    %[[TMP30:.+]] = getelementptr inbounds float, ptr %[[TMP0:.+]], i32 %[[TMP29:.+]]
// CHECK-NEXT:    store float 4.200000e+01, ptr %[[TMP30:.+]], align 4
// CHECK-NEXT:    br label %[[OMP_REGION_CONT:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_REGION_CONT]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_INC30:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_INC30]]:
// CHECK-NEXT:    %[[OMP_TILE0_NEXT35:.+]] = add nuw i32 %[[OMP_TILE0_IV33:.+]], 1
// CHECK-NEXT:    br label %[[OMP_TILE0_HEADER27:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_EXIT31]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_AFTER32:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_AFTER32]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_INC:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_INC]]:
// CHECK-NEXT:    %[[OMP_FLOOR0_NEXT:.+]] = add nuw i32 %[[OMP_FLOOR0_IV:.+]], 1
// CHECK-NEXT:    br label %[[OMP_FLOOR0_HEADER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_AFTER25:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_AFTER25]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_AFTER]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_INC16:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_INC16]]:
// CHECK-NEXT:    %[[OMP_TILE0_NEXT21:.+]] = add nuw i32 %[[OMP_TILE0_IV19:.+]], 1
// CHECK-NEXT:    br label %[[OMP_TILE0_HEADER13:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_EXIT17]]:
// CHECK-NEXT:    br label %[[OMP_TILE0_AFTER18:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_TILE0_AFTER18]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_INC6:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_INC6]]:
// CHECK-NEXT:    %[[OMP_FLOOR0_NEXT11:.+]] = add nuw i32 %[[OMP_FLOOR0_IV9:.+]], 1
// CHECK-NEXT:    br label %[[OMP_FLOOR0_HEADER3:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_EXIT7]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_AFTER8:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_AFTER8]]:
// CHECK-NEXT:    br label %[[OMP_FLOOR0_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_FLOOR0_AFTER]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_AFTER:.+]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_AFTER]]:
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }


// CHECK:       !llvm.module.flags = !{!0}
// CHECK-EMPTY:
// CHECK-NEXT:  !0 = !{i32 2, !"Debug Info Version", i32 3}
