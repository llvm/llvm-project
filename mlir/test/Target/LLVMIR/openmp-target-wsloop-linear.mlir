// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa",
                   omp.is_gpu = true, omp.is_target_device = true} {
  llvm.func @wsloop_linear_target(%x : !llvm.ptr) attributes {
    omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>
  } {
    %lb = llvm.mlir.constant(0 : i32) : i32
    %ub = llvm.mlir.constant(9 : i32) : i32
    %step = llvm.mlir.constant(1 : i32) : i32
    omp.wsloop linear(%x : !llvm.ptr = %step : i32) {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        omp.yield
      }
    } {linear_var_types = [i32]}
    llvm.return
  }
}

// CHECK-LABEL: define {{(protected )?}}void @wsloop_linear_target
// CHECK: %[[P_LASTITER:.*]] = alloca i32
// CHECK: store i32 0, ptr %[[P_LASTITER]]
// CHECK: call void @__kmpc_for_static_loop_4u({{.*}}, ptr @wsloop_linear_target..omp_wsloop
// CHECK: %[[LASTITER_VAL:.*]] = load i32, ptr %[[P_LASTITER]]
// CHECK: %[[IS_LASTITER:.*]] = icmp ne i32 %[[LASTITER_VAL]], 0
// CHECK: br i1 %[[IS_LASTITER]], label %[[LINEAR_UPDATE_BLOCK:.*]], label %[[LINEAR_SKIP_BLOCK:.*]]

// CHECK-LABEL: define internal void @wsloop_linear_target..omp_wsloop(
// CHECK-SAME: ptr %[[LASTITER_ARG:.*]], i32 %[[LOOP_CNT_ARG:.*]])
// CHECK: %[[CMP:.*]] = icmp eq i32 %[[LOOP_CNT_ARG]], 9
// CHECK: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
// CHECK: store i32 %[[ZEXT]], ptr %[[LASTITER_ARG]]
