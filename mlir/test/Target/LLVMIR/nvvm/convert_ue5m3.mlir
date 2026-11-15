// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// -----

// CHECK-LABEL: @convert_f32x2_to_ue5m3x2
llvm.func @convert_f32x2_to_ue5m3x2(%srcA : f32, %srcB : f32, %scale_factor : i8) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rn(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB rnd = <rn> : i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB rnd = <rn> sat = <satfinite> : i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = zext i8 %{{.*}} to i16
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rn.scale.n1.ue8m0(float %{{.*}}, float %{{.*}}, i16 %{{.*}})
  %res3 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB, %scale_factor rnd = <rn> : i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rn.satfinite.scale.n1.ue8m0(float %{{.*}}, float %{{.*}}, i16 %{{.*}})
  %res4 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB, %scale_factor rnd = <rn> sat = <satfinite> : i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rz(float %{{.*}}, float %{{.*}})
  %res5 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB rnd = <rz> : i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rz.satfinite(float %{{.*}}, float %{{.*}})
  %res6 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB rnd = <rz> sat = <satfinite> : i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rz.scale.n1.ue8m0(float %{{.*}}, float %{{.*}}, i16 %{{.*}})
  %res7 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB, %scale_factor rnd = <rz> : i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rz.satfinite.scale.n1.ue8m0(float %{{.*}}, float %{{.*}}, i16 %{{.*}})
  %res8 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB, %scale_factor rnd = <rz> sat = <satfinite> : i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rp(float %{{.*}}, float %{{.*}})
  %res9 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB rnd = <rp> : i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.ue5m3x2.rp.satfinite(float %{{.*}}, float %{{.*}})
  %res10 = nvvm.convert.f32x2.to.f8x2 %srcA, %srcB rnd = <rp> sat = <satfinite> : i16 (f8E5M3FNU)
  llvm.return
}

// CHECK-LABEL: @convert_f16x2_to_ue5m3x2
llvm.func @convert_f16x2_to_ue5m3x2(%src : vector<2xf16>, %scale_factor : i8) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rn(<2 x half> %{{.*}})
  %res1 = nvvm.convert.f16x2.to.f8x2 %src rnd = <rn> : vector<2xf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rn.satfinite(<2 x half> %{{.*}})
  %res2 = nvvm.convert.f16x2.to.f8x2 %src rnd = <rn> sat = <satfinite> : vector<2xf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rn.scale.n1.ue8m0(<2 x half> %{{.*}}, i16 %{{.*}})
  %res3 = nvvm.convert.f16x2.to.f8x2 %src, %scale_factor rnd = <rn> : vector<2xf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rn.satfinite.scale.n1.ue8m0(<2 x half> %{{.*}}, i16 %{{.*}})
  %res4 = nvvm.convert.f16x2.to.f8x2 %src, %scale_factor rnd = <rn> sat = <satfinite> : vector<2xf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rz(<2 x half> %{{.*}})
  %res5 = nvvm.convert.f16x2.to.f8x2 %src rnd = <rz> : vector<2xf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rz.satfinite(<2 x half> %{{.*}})
  %res6 = nvvm.convert.f16x2.to.f8x2 %src rnd = <rz> sat = <satfinite> : vector<2xf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rz.scale.n1.ue8m0(<2 x half> %{{.*}}, i16 %{{.*}})
  %res7 = nvvm.convert.f16x2.to.f8x2 %src, %scale_factor rnd = <rz> : vector<2xf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rz.satfinite.scale.n1.ue8m0(<2 x half> %{{.*}}, i16 %{{.*}})
  %res8 = nvvm.convert.f16x2.to.f8x2 %src, %scale_factor rnd = <rz> sat = <satfinite> : vector<2xf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rp(<2 x half> %{{.*}})
  %res9 = nvvm.convert.f16x2.to.f8x2 %src rnd = <rp> : vector<2xf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.ue5m3x2.rp.satfinite(<2 x half> %{{.*}})
  %res10 = nvvm.convert.f16x2.to.f8x2 %src rnd = <rp> sat = <satfinite> : vector<2xf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// CHECK-LABEL: @convert_bf16x2_to_ue5m3x2
llvm.func @convert_bf16x2_to_ue5m3x2(%src : vector<2xbf16>, %scale_factor : i8) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rn(<2 x bfloat> %{{.*}})
  %res1 = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rn> : vector<2xbf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rn.satfinite(<2 x bfloat> %{{.*}})
  %res2 = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rn> sat = <satfinite> : vector<2xbf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rn.scale.n1.ue8m0(<2 x bfloat> %{{.*}}, i16 %{{.*}})
  %res3 = nvvm.convert.bf16x2.to.f8x2 %src, %scale_factor rnd = <rn> : vector<2xbf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rn.satfinite.scale.n1.ue8m0(<2 x bfloat> %{{.*}}, i16 %{{.*}})
  %res4 = nvvm.convert.bf16x2.to.f8x2 %src, %scale_factor rnd = <rn> sat = <satfinite> : vector<2xbf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rz(<2 x bfloat> %{{.*}})
  %res5 = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rz> : vector<2xbf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rz.satfinite(<2 x bfloat> %{{.*}})
  %res6 = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rz> sat = <satfinite> : vector<2xbf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rz.scale.n1.ue8m0(<2 x bfloat> %{{.*}}, i16 %{{.*}})
  %res7 = nvvm.convert.bf16x2.to.f8x2 %src, %scale_factor rnd = <rz> : vector<2xbf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rz.satfinite.scale.n1.ue8m0(<2 x bfloat> %{{.*}}, i16 %{{.*}})
  %res8 = nvvm.convert.bf16x2.to.f8x2 %src, %scale_factor rnd = <rz> sat = <satfinite> : vector<2xbf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rp(<2 x bfloat> %{{.*}})
  %res9 = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rp> : vector<2xbf16> -> i16 (f8E5M3FNU)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.ue5m3x2.rp.satfinite(<2 x bfloat> %{{.*}})
  %res10 = nvvm.convert.bf16x2.to.f8x2 %src rnd = <rp> sat = <satfinite> : vector<2xbf16> -> i16 (f8E5M3FNU)
  llvm.return
}

// -----

// CHECK-LABEL: @convert_ue5m3x2_to_f16x2
llvm.func @convert_ue5m3x2_to_f16x2(%src : vector<2xi8>) {
  // CHECK: %[[res1:.*]] = bitcast <2 x i8> %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x half> @llvm.nvvm.ue5m3x2.to.f16x2.rn(i16 %[[res1]])
  %res1 = nvvm.convert.f8x2.to.f16x2 %src : vector<2xi8> (f8E5M3FNU) -> vector<2xf16>
  llvm.return
}

// CHECK-LABEL: @convert_ue5m3x2_to_bf16x2
llvm.func @convert_ue5m3x2_to_bf16x2(%src : vector<2xi8>, %scale_factor : i16) {
  // CHECK: %[[res1:.*]] = bitcast <2 x i8> %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ue5m3x2.to.bf16x2.rn(i16 %[[res1]])
  %res1 = nvvm.convert.f8x2.to.bf16x2 %src : vector<2xi8> (f8E5M3FNU) -> vector<2xbf16>
  // CHECK: %[[res2:.*]] = bitcast <2 x i8> %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ue5m3x2.to.bf16x2.rn.satfinite(i16 %[[res2]])
  %res2 = nvvm.convert.f8x2.to.bf16x2 %src sat = <satfinite> : vector<2xi8> (f8E5M3FNU) -> vector<2xbf16>
  // CHECK: %[[res3:.*]] = bitcast <2 x i8> %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ue5m3x2.to.bf16x2.rn.scale.n2.ue8m0(i16 %[[res3]], i16 %{{.*}})
  %res3 = nvvm.convert.f8x2.to.bf16x2 %src scale_factor = %scale_factor : vector<2xi8> (f8E5M3FNU) -> vector<2xbf16>
  // CHECK: %[[res4:.*]] = bitcast <2 x i8> %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x bfloat> @llvm.nvvm.ue5m3x2.to.bf16x2.rn.satfinite.scale.n2.ue8m0(i16 %[[res4]], i16 %{{.*}})
  %res4 = nvvm.convert.f8x2.to.bf16x2 %src scale_factor = %scale_factor sat = <satfinite> : vector<2xi8> (f8E5M3FNU) -> vector<2xbf16>
  llvm.return
}
