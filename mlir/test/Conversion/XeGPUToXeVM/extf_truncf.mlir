// RUN: mlir-opt --split-input-file -convert-xegpu-to-xevm %s | FileCheck %s

// Micro-scaling extf/truncf between the MX narrow floats (f8E5M2, f8E4M3FN,
// f4E2M1FN) and f16/bf16 are lowered to xevm.extf / xevm.truncf.

// CHECK-LABEL: gpu.func @extf_e2m1_bf16
// CHECK-SAME: (%[[ARG0:.*]]: vector<16xf4E2M1FN>)
gpu.module @extf_e2m1_bf16 [#xevm.target<chip = "cri">] {
  gpu.func @extf_e2m1_bf16(%a: vector<16xf4E2M1FN>) kernel {
    // CHECK: %[[I4:.*]] = vector.bitcast %[[ARG0]] : vector<16xf4E2M1FN> to vector<16xi4>
    // CHECK: %[[I8:.*]] = vector.bitcast %[[I4]] : vector<16xi4> to vector<8xi8>
    // CHECK: %{{.*}} = xevm.extf %[[I8]] {src_etype = e2m1, dst_etype = bf16} : (vector<8xi8>) -> vector<16xbf16>
    %r = arith.extf %a : vector<16xf4E2M1FN> to vector<16xbf16>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @extf_e2m1_f16
// CHECK-SAME: (%[[ARG0:.*]]: vector<16xf4E2M1FN>)
gpu.module @extf_e2m1_f16 [#xevm.target<chip = "cri">] {
  gpu.func @extf_e2m1_f16(%a: vector<16xf4E2M1FN>) kernel {
    // CHECK: %[[I4:.*]] = vector.bitcast %[[ARG0]] : vector<16xf4E2M1FN> to vector<16xi4>
    // CHECK: %[[I8:.*]] = vector.bitcast %[[I4]] : vector<16xi4> to vector<8xi8>
    // CHECK: %{{.*}} = xevm.extf %[[I8]] {src_etype = e2m1, dst_etype = f16} : (vector<8xi8>) -> vector<16xf16>
    %r = arith.extf %a : vector<16xf4E2M1FN> to vector<16xf16>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @extf_bf8_f16
// CHECK-SAME: (%[[ARG0:.*]]: vector<16xf8E5M2>)
gpu.module @extf_bf8_f16 [#xevm.target<chip = "cri">] {
  gpu.func @extf_bf8_f16(%a: vector<16xf8E5M2>) kernel {
    // CHECK: %[[I8:.*]] = vector.bitcast %[[ARG0]] : vector<16xf8E5M2> to vector<16xi8>
    // CHECK: %{{.*}} = xevm.extf %[[I8]] {src_etype = bf8, dst_etype = f16} : (vector<16xi8>) -> vector<16xf16>
    %r = arith.extf %a : vector<16xf8E5M2> to vector<16xf16>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @extf_f8_bf16
// CHECK-SAME: (%[[ARG0:.*]]: vector<16xf8E4M3FN>)
gpu.module @extf_f8_bf16 [#xevm.target<chip = "cri">] {
  gpu.func @extf_f8_bf16(%a: vector<16xf8E4M3FN>) kernel {
    // CHECK: %[[I8:.*]] = vector.bitcast %[[ARG0]] : vector<16xf8E4M3FN> to vector<16xi8>
    // CHECK: %{{.*}} = xevm.extf %[[I8]] {src_etype = f8, dst_etype = bf16} : (vector<16xi8>) -> vector<16xbf16>
    %r = arith.extf %a : vector<16xf8E4M3FN> to vector<16xbf16>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @truncf_f16_e2m1
// CHECK-SAME: (%[[ARG0:.*]]: vector<16xf16>)
gpu.module @truncf_f16_e2m1 [#xevm.target<chip = "cri">] {
  gpu.func @truncf_f16_e2m1(%a: vector<16xf16>) kernel {
    // CHECK: %[[I8:.*]] = xevm.truncf %[[ARG0]] {src_etype = f16, dst_etype = e2m1} : (vector<16xf16>) -> vector<8xi8>
    // CHECK: %{{.*}} = vector.bitcast %[[I8]] : vector<8xi8> to vector<16xi4>
    %r = arith.truncf %a : vector<16xf16> to vector<16xf4E2M1FN>
    gpu.return
  }
}

// -----

// CHECK-LABEL: gpu.func @truncf_bf16_f8
// CHECK-SAME: (%[[ARG0:.*]]: vector<16xbf16>)
gpu.module @truncf_bf16_f8 [#xevm.target<chip = "cri">] {
  gpu.func @truncf_bf16_f8(%a: vector<16xbf16>) kernel {
    // CHECK: %{{.*}} = xevm.truncf %[[ARG0]] {src_etype = bf16, dst_etype = f8} : (vector<16xbf16>) -> vector<16xi8>
    %r = arith.truncf %a : vector<16xbf16> to vector<16xf8E4M3FN>
    gpu.return
  }
}

// -----

// Plain float extensions/truncations are not micro-scaling and must be left
// untouched for the regular arith-to-LLVM lowering.

// CHECK-LABEL: gpu.func @extf_passthrough
gpu.module @extf_passthrough [#xevm.target<chip = "cri">] {
  gpu.func @extf_passthrough(%a: vector<16xf16>) kernel {
    // CHECK: %{{.*}} = arith.extf %{{.*}} : vector<16xf16> to vector<16xf32>
    // CHECK-NOT: xevm.extf
    %r = arith.extf %a : vector<16xf16> to vector<16xf32>
    gpu.return
  }
}
