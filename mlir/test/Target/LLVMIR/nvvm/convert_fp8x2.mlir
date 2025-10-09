// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// -----

// CHECK-LABEL: @convert_f32x2_to_f8x2_e4m3
llvm.func @convert_f32x2_to_f8x2_e4m3(%srcA : f32, %srcB : f32) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e4m3x2.rn(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<satfinite>} : i16 (f8E4M3FN)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e4m3x2.rn.relu(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {relu = true, rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<satfinite>} : i16 (f8E4M3FN)
  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_f8x2_e5m2
llvm.func @convert_f32x2_to_f8x2_e5m2(%srcA : f32, %srcB : f32) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e5m2x2.rn(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<satfinite>} : i16 (f8E5M2)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e5m2x2.rn.relu(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {relu = true, rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<satfinite>} : i16 (f8E5M2)
  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_f8x2_ue8m0
llvm.func @convert_f32x2_to_f8x2_ue8m0(%srcA : f32, %srcB : f32) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue8m0x2.rz(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>} : i16 (f8E8M0FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue8m0x2.rp(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rp>} : i16 (f8E8M0FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue8m0x2.rz.satfinite(float %{{.*}}, float %{{.*}})
  %res3 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<satfinite>} : i16 (f8E8M0FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue8m0x2.rp.satfinite(float %{{.*}}, float %{{.*}})
  %res4 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<satfinite>} : i16 (f8E8M0FNU)
  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_f8x2_vector_return
llvm.func @convert_f32x2_to_f8x2_vector_return(%srcA : f32, %srcB : f32) {
  // CHECK: %[[res1:.*]] = call i16 @llvm.nvvm.ff.to.e4m3x2.rn(float %{{.*}}, float %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %[[res1]] to <2 x i8>
  %res1 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<satfinite>} : vector<2xi8> (f8E4M3FN)
  // CHECK: %[[res2:.*]] = call i16 @llvm.nvvm.ff.to.e4m3x2.rn.relu(float %{{.*}}, float %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %[[res2]] to <2 x i8>
  %res2 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB {relu = true, rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<satfinite>} : vector<2xi8> (f8E4M3FN)
  llvm.return
}

// -----

// CHECK-LABEL: @convert_f16x2_to_f8x2_e4m3
llvm.func @convert_f16x2_to_f8x2_e4m3(%src : vector<2xf16>) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.e4m3x2.rn(<2 x half> %{{.*}})
  %res1 = nvvm.convert.f16x2.to.f8x2 %src : vector<2xf16> -> i16 (f8E4M3FN)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.e4m3x2.rn.relu(<2 x half> %{{.*}})
  %res2 = nvvm.convert.f16x2.to.f8x2 %src {relu = true} : vector<2xf16> -> i16 (f8E4M3FN)
  llvm.return
}

// CHECK-LABEL: @convert_f16x2_to_f8x2_e5m2
llvm.func @convert_f16x2_to_f8x2_e5m2(%src : vector<2xf16>) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.e5m2x2.rn(<2 x half> %{{.*}})
  %res1 = nvvm.convert.f16x2.to.f8x2 %src : vector<2xf16> -> i16 (f8E5M2)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.e5m2x2.rn.relu(<2 x half> %{{.*}})
  %res2 = nvvm.convert.f16x2.to.f8x2 %src {relu = true} : vector<2xf16> -> i16 (f8E5M2)
  llvm.return
}

// CHECK-LABEL: @convert_f16x2_to_f8x2_vector_return
llvm.func @convert_f16x2_to_f8x2_vector_return(%src : vector<2xf16>) {
  // CHECK: %[[res1:.*]] = call i16 @llvm.nvvm.f16x2.to.e4m3x2.rn(<2 x half> %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %[[res1]] to <2 x i8>
  %res1 = nvvm.convert.f16x2.to.f8x2 %src : vector<2xf16> -> vector<2xi8> (f8E4M3FN)
  // CHECK: %[[res2:.*]] = call i16 @llvm.nvvm.f16x2.to.e5m2x2.rn(<2 x half> %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %[[res2]] to <2 x i8>
  %res2 = nvvm.convert.f16x2.to.f8x2 %src : vector<2xf16> -> vector<2xi8> (f8E5M2)
  llvm.return
}

// -----

// CHECK-LABEL: @convert_bf16x2_to_f8x2_ue8m0
llvm.func @convert_bf16x2_to_f8x2_ue8m0(%src : vector<2xbf16>) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rz(<2 x bfloat> %{{.*}})
  %res1 = nvvm.convert.bf16x2.to.f8x2 %src {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xbf16> -> i16 (f8E8M0FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rp(<2 x bfloat> %{{.*}})
  %res2 = nvvm.convert.bf16x2.to.f8x2 %src {rnd = #nvvm.fp_rnd_mode<rp>} : vector<2xbf16> -> i16 (f8E8M0FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rz.satfinite(<2 x bfloat> %{{.*}})
  %res3 = nvvm.convert.bf16x2.to.f8x2 %src {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16> -> i16 (f8E8M0FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rp.satfinite(<2 x bfloat> %{{.*}})
  %res4 = nvvm.convert.bf16x2.to.f8x2 %src {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16> -> i16 (f8E8M0FNU)
  llvm.return
}

// CHECK-LABEL: @convert_bf16x2_to_f8x2_vector_return
llvm.func @convert_bf16x2_to_f8x2_vector_return(%src : vector<2xbf16>) {
  // CHECK: %[[res1:.*]] = call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rz(<2 x bfloat> %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %[[res1]] to <2 x i8>
  %res1 = nvvm.convert.bf16x2.to.f8x2 %src {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xbf16> -> vector<2xi8> (f8E8M0FNU)
  // CHECK: %[[res2:.*]] = call i16 @llvm.nvvm.bf16x2.to.ue8m0x2.rp.satfinite(<2 x bfloat> %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %[[res2]] to <2 x i8>
  %res2 = nvvm.convert.bf16x2.to.f8x2 %src {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<satfinite>} : vector<2xbf16> -> vector<2xi8> (f8E8M0FNU)
  llvm.return
}
