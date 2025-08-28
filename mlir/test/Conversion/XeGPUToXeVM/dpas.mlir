// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

#sg_map_a_f16 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
#sg_map_b_f16 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>
#sg_map_c_f32 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>

gpu.module @load_store_check {
    // CHECK-LABEL: func.func @dpas(
    // CHECK-SAME: %[[ARG0:.*]]: vector<8xf16>, %[[ARG1:.*]]: vector<16xf16>, %[[ARG2:.*]]: vector<8xf32>
    func.func @dpas(%a_loaded: vector<8xf16>, %b_loaded: vector<16xf16>, %c_loaded: vector<8xf32>) -> vector<8xf32> {
        // Loads are checked in a separate test.
        // CHECK: %[[D:.*]] = xevm.mma %[[ARG0]], %[[ARG1]], %[[ARG2]] {shape = <m = 8, n = 16, k = 16>, types = <d = f32, a = f16, b = f16, c = f32>}
        // CHECK-SAME:    : (vector<8xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
        %d = xegpu.dpas %a_loaded, %b_loaded, %c_loaded {a_layout = #sg_map_a_f16, b_layout = #sg_map_b_f16, c_layout = #sg_map_c_f32}
            : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
        return %d : vector<8xf32>
    }
}
