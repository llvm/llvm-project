// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @convert_f32x2_to_f16x2_rn
llvm.func @convert_f32x2_to_f16x2_rn(%srcA : f32, %srcB : f32) {
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rn(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xf16>
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<satfinite>} : vector<2xf16>
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rn.relu(float %{{.*}}, float %{{.*}})
  %res3 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>, relu = true} : vector<2xf16>
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rn.relu.satfinite(float %{{.*}}, float %{{.*}})
  %res4 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>, relu = true, sat = #nvvm.sat_mode<satfinite>} : vector<2xf16>
  
  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_f16x2_rz
llvm.func @convert_f32x2_to_f16x2_rz(%srcA : f32, %srcB : f32) {
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rz(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf16>
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rz.satfinite(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<satfinite>} : vector<2xf16>
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rz.relu(float %{{.*}}, float %{{.*}})
  %res3 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>, relu = true} : vector<2xf16>
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rz.relu.satfinite(float %{{.*}}, float %{{.*}})
  %res4 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>, relu = true, sat = #nvvm.sat_mode<satfinite>} : vector<2xf16>

  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_f16x2_rs_stochastic
llvm.func @convert_f32x2_to_f16x2_rs_stochastic(%srcA : f32, %srcB : f32, %rbits : i32) {
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rs(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res1 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB, %rbits {rnd = #nvvm.fp_rnd_mode<rs>} : vector<2xf16>
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rs.relu(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res2 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB, %rbits {relu = true, rnd = #nvvm.fp_rnd_mode<rs>} : vector<2xf16>
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rs.satfinite(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res3 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB, %rbits {rnd = #nvvm.fp_rnd_mode<rs>, sat = #nvvm.sat_mode<satfinite>} : vector<2xf16>
  // CHECK: %{{.*}} = call <2 x half> @llvm.nvvm.ff2f16x2.rs.relu.satfinite(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res4 = nvvm.convert.f32x2.to.f16x2 %srcA, %srcB, %rbits {relu = true, rnd = #nvvm.fp_rnd_mode<rs>, sat = #nvvm.sat_mode<satfinite>} : vector<2xf16>

  llvm.return
}

// -----

// CHECK-LABEL: @convert_f32x2_to_bf16x2_rn
llvm.func @convert_f32x2_to_bf16x2_rn(%srcA : f32, %srcB : f32) {
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xbf16>
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16>
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn.relu(float %{{.*}}, float %{{.*}})
  %res3 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>, relu = true} : vector<2xbf16>
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn.relu.satfinite(float %{{.*}}, float %{{.*}})
  %res4 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>, relu = true, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16>

  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_bf16x2_rz
llvm.func @convert_f32x2_to_bf16x2_rz(%srcA : f32, %srcB : f32) {
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xbf16>
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz.satfinite(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16>
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz.relu(float %{{.*}}, float %{{.*}})
  %res3 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>, relu = true} : vector<2xbf16>
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz.relu.satfinite(float %{{.*}}, float %{{.*}})
  %res4 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>, relu = true, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16>

  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_bf16x2_rs_stochastic
llvm.func @convert_f32x2_to_bf16x2_rs_stochastic(%srcA : f32, %srcB : f32, %rbits : i32) {
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res1 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB, %rbits {rnd = #nvvm.fp_rnd_mode<rs>} : vector<2xbf16>
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.relu(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res2 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB, %rbits {relu = true, rnd = #nvvm.fp_rnd_mode<rs>} : vector<2xbf16>
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.satfinite(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res3 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB, %rbits {rnd = #nvvm.fp_rnd_mode<rs>, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16>
  // CHECK: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rs.relu.satfinite(float %{{.*}}, float %{{.*}}, i32 %{{.*}})
  %res4 = nvvm.convert.f32x2.to.bf16x2 %srcA, %srcB, %rbits {relu = true, rnd = #nvvm.fp_rnd_mode<rs>, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16>

  llvm.return
}
