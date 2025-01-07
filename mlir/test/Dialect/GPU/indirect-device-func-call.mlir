// RUN: mlir-opt -test-gpu-rewrite -convert-func-to-llvm %s | FileCheck %s

gpu.module @kernels {
    // CHECK-LABEL: @hello
    // CHECK-SAME: %[[ARG0:.*]]: f32
    func.func @hello(%arg0 : f32) {
        %tid_x = gpu.thread_id x
        %csti8 = arith.constant 2 : i8
        gpu.printf "Hello from %lld, %d, %f\n" %tid_x, %csti8, %arg0  : index, i8, f32
        return
    }
    // CHECK-LABEL: @hello_indirect
    gpu.func @hello_indirect() kernel { 
        %cstf32 = arith.constant 3.0 : f32
        // CHECK: %[[DEVICE_FUNC_ADDR:.*]] = llvm.mlir.addressof @hello : !llvm.ptr
        %func_ref = func.constant @hello : (f32) -> ()
        // CHECK: llvm.call %[[DEVICE_FUNC_ADDR]](%{{.*}}) : !llvm.ptr, (f32) -> ()
        func.call_indirect %func_ref(%cstf32) : (f32) -> ()
        gpu.return
    }
}
