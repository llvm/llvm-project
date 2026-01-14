// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @test_kernel {
    // CHECK-LABEL: func.func @dpas(
    // CHECK-SAME: %[[ARG0:.*]]: vector<8xf16>, %[[ARG1:.*]]: vector<16xf16>, %[[ARG2:.*]]: vector<8xf32>
    func.func @dpas(%a_loaded: vector<8xf16>, %b_loaded: vector<16xf16>, %c_loaded: vector<8xf32>) -> vector<8xf32> {
        // Loads are checked in a separate test.
        // CHECK: %[[D:.*]] = xevm.mma %[[ARG0]], %[[ARG1]], %[[ARG2]] {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>}
        // CHECK-SAME:    : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %d = xegpu.dpas %a_loaded, %b_loaded, %c_loaded
            : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
        return %d : vector<8xf32>
    }
}
