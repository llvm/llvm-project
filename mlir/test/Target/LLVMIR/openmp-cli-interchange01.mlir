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
// CHECK-SAME:    ptr %[[VAL_0:.*]], i32 %[[VAL_1:.*]], i32 %[[VAL_2:.*]]) {
// CHECK-NEXT:    br label  %[[OMP_INTERCHANGE0_PREHEADER:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE0_PREHEADER]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_HEADER:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE0_HEADER]]:
// CHECK-NEXT:    %[[OMP_INTERCHANGE0_IV:.*]] = phi i32 [ 0, %[[OMP_INTERCHANGE0_PREHEADER]] ], [ %[[OMP_INTERCHANGE0_NEXT:.*]], %[[OMP_INTERCHANGE0_INC:.*]] ]
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_COND:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE0_COND]]:
// CHECK-NEXT:    %[[OMP_INTERCHANGE0_CMP:.*]] = icmp ult i32 %[[OMP_INTERCHANGE0_IV]], %[[VAL_2:.*]]
// CHECK-NEXT:    br i1 %[[OMP_INTERCHANGE0_CMP]], label %[[OMP_INTERCHANGE0_BODY:.*]], label %[[OMP_INTERCHANGE0_EXIT:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE0_BODY]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_PREHEADER:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE1_PREHEADER]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_HEADER:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE1_HEADER]]:
// CHECK-NEXT:    %[[OMP_INTERCHANGE1_IV:.*]] = phi i32 [ 0, %[[OMP_INTERCHANGE1_PREHEADER]] ], [ %[[OMP_INTERCHANGE1_NEXT:.*]], %[[OMP_INTERCHANGE1_INC:.*]] ]
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_COND:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE1_COND]]:
// CHECK-NEXT:    %[[OMP_INTERCHANGE1_CMP:.*]] = icmp ult i32 %[[OMP_INTERCHANGE1_IV]], %[[VAL_1:.*]]
// CHECK-NEXT:    br i1 %[[OMP_INTERCHANGE1_CMP]], label %[[OMP_INTERCHANGE1_BODY:.*]], label %[[OMP_INTERCHANGE1_EXIT:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE1_BODY]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_BODY4:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_BODY4]]:
// CHECK-NEXT:    br label %[[OMP_LOOP_REGION12:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_LOOP_REGION12]]:
// CHECK-NEXT:    %[[VAL_3:.*]] = getelementptr inbounds float, ptr %[[VAL_0:.*]], i32 %[[OMP_INTERCHANGE1_IV]]
// CHECK-NEXT:    store float 4.200000e+01, ptr %[[VAL_3]], align 4
// CHECK-NEXT:    br label %[[OMP_REGION_CONT11:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_REGION_CONT11]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_INC]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE1_INC]]:
// CHECK-NEXT:    %[[OMP_INTERCHANGE1_NEXT]] = add nuw i32 %[[OMP_INTERCHANGE1_IV]], 1
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_HEADER]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE1_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE1_AFTER:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE1_AFTER]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_INC]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE0_INC]]:
// CHECK-NEXT:    %[[OMP_INTERCHANGE0_NEXT]] = add nuw i32 %[[OMP_INTERCHANGE0_IV]], 1
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_HEADER]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE0_EXIT]]:
// CHECK-NEXT:    br label %[[OMP_INTERCHANGE0_AFTER:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_INTERCHANGE0_AFTER]]:
// CHECK-NEXT:    br label %[[OMP_OMP_LOOP_AFTER:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[OMP_OMP_LOOP_AFTER]]:
// CHECK-NEXT:    ret void

