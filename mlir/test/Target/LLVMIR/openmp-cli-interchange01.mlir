// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --enable-var-scope


llvm.func @interchange_loop(%baseptr: !llvm.ptr, %tc1: i32, %tc2: i32) -> () {
  %cli_outer = omp.new_cli
  %cli_inner = omp.new_cli
  omp.canonical_loop(%cli_outer) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%cli_inner) %iv2 : i32 in range(%tc2) {
      %ptr = llvm.getelementptr inbounds %baseptr[%iv1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %val = llvm.mlir.constant(42.0 : f32) : f32
      llvm.store %val, %ptr : f32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  omp.interchange <- (%cli_outer, %cli_inner) {permutation = [2 : i32, 1 : i32]}
  llvm.return
}
// CHECK-LABEL: define void @interchange_loop(
// CHECK-SAME: ptr [[TMP0:%.*]], i32 [[TMP1:%.*]], i32 [[TMP2:%.*]]) {
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_PREHEADER:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_PREHEADER]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_PREHEADER:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_HEADER:.*]]:
// CHECK-NEXT:    [[OMP_OMP_LOOP_IV:%.*]] = phi i32 [ [[OMP_OMP_LOOP_NEXT:%.*]], %[[OMP_OMP_LOOP_INC:.*]] ]
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_COND:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_COND]]:
// CHECK-NEXT:    [[OMP_OMP_LOOP_CMP:%.*]] = icmp ult i32 [[OMP_INTERCHANGE1_IV:%.*]], [[TMP1]]
// CHECK-NEXT:    br i1 [[OMP_OMP_LOOP_CMP]], label %[[OMP_OMP_LOOP_BODY:.*]], label %[[OMP_OMP_LOOP_EXIT:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_BODY]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_REGION:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_LOOP_REGION]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_PREHEADER1:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_PREHEADER1]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_HEADER2:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_HEADER2]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_COND3:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_COND3]]:
// CHECK-NEXT:    [[OMP_OMP_LOOP_CMP9:%.*]] = icmp ult i32 [[OMP_INTERCHANGE0_IV:%.*]], [[TMP2]]
// CHECK-NEXT:    br i1 [[OMP_OMP_LOOP_CMP9]], label %[[OMP_OMP_LOOP_BODY4:.*]], label %[[OMP_OMP_LOOP_EXIT6:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE0_PREHEADER]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_HEADER:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE0_HEADER]]:
// CHECK-NEXT:    [[OMP_INTERCHANGE0_IV]] = phi i32 [ 0, %[[OMP_INTERCHANGE0_PREHEADER]] ], [ [[OMP_INTERCHANGE0_NEXT:%.*]], %[[OMP_INTERCHANGE0_INC:.*]] ]
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_COND:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE0_COND]]:
// CHECK-NEXT:    [[OMP_INTERCHANGE0_CMP:%.*]] = icmp ult i32 [[OMP_INTERCHANGE0_IV]], [[TMP2]]
// CHECK-NEXT:    br i1 [[OMP_INTERCHANGE0_CMP]], label %[[OMP_INTERCHANGE0_BODY:.*]], label %[[OMP_INTERCHANGE0_EXIT:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE0_BODY]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_PREHEADER:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE1_PREHEADER]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_HEADER:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE1_HEADER]]:
// CHECK-NEXT:    [[OMP_INTERCHANGE1_IV]] = phi i32 [ 0, %[[OMP_INTERCHANGE1_PREHEADER]] ], [ [[OMP_INTERCHANGE1_NEXT:%.*]], %[[OMP_INTERCHANGE1_INC:.*]] ]
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_COND:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE1_COND]]:
// CHECK-NEXT:    [[OMP_INTERCHANGE1_CMP:%.*]] = icmp ult i32 [[OMP_INTERCHANGE1_IV]], [[TMP1]]
// CHECK-NEXT:    br i1 [[OMP_INTERCHANGE1_CMP]], label %[[OMP_INTERCHANGE1_BODY:.*]], label %[[OMP_INTERCHANGE1_EXIT:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE1_BODY]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_BODY4]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_BODY4]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_REGION12:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_LOOP_REGION12]]:
// CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds float, ptr [[TMP0]], i32 [[OMP_INTERCHANGE1_IV]]
// CHECK-NEXT:    store float 4.200000e+01, ptr [[TMP4]], align 4
// CHECK-NEXT:    br label %[[OMP_REGION_CONT11:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_REGION_CONT11]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_INC]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE1_INC]]:
// CHECK-NEXT:    [[OMP_INTERCHANGE1_NEXT]] = add nuw i32 [[OMP_INTERCHANGE1_IV]], 1
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_HEADER]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE1_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_AFTER:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE1_AFTER]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_INC]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE0_INC]]:
// CHECK-NEXT:    [[OMP_INTERCHANGE0_NEXT]] = add nuw i32 [[OMP_INTERCHANGE0_IV]], 1
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_HEADER]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE0_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_AFTER:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_INTERCHANGE0_AFTER]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_AFTER:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_EXIT6]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_AFTER7:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_AFTER7]]:
// CHECK-NEXT:    br label %[[OMP_REGION_CONT:.*]]
// CHECK-EMPTY:
// CHECK:       [[OMP_REGION_CONT]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_INC]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_INC]]:
// CHECK-NEXT:    [[OMP_OMP_LOOP_NEXT]] = add nuw i32 [[OMP_INTERCHANGE1_IV]], 1
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_HEADER]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_AFTER]]
// CHECK-EMPTY:
// CHECK:       [[OMP_OMP_LOOP_AFTER]]:
// CHECK-NEXT:    ret void

