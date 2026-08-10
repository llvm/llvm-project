// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @convert_f32x2_to_f4x2_e2m1
llvm.func @convert_f32x2_to_f4x2_e2m1(%srcA : f32, %srcB : f32) {
  // CHECK: %[[res1:.*]] = call i16 @llvm.nvvm.ff.to.e2m1x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  // CHECK-NEXT: %{{.*}} = trunc i16 %[[res1]] to i8
  %res1 = nvvm.convert.f32x2.to.f4x2 %srcA, %srcB : i8 (f4E2M1FN)
  // CHECK: %[[res2:.*]] = call i16 @llvm.nvvm.ff.to.e2m1x2.rn.relu.satfinite(float %{{.*}}, float %{{.*}})
  // CHECK-NEXT: %{{.*}} = trunc i16 %[[res2]] to i8
  %res2 = nvvm.convert.f32x2.to.f4x2 %srcA, %srcB {relu = true} : i8 (f4E2M1FN)
  llvm.return
}

// -----

// CHECK-LABEL: @convert_f16x2_to_f4x2
llvm.func @convert_f16x2_to_f4x2(%srcA : vector<2xf16>) {
  // CHECK: %[[res1:.*]] = call i16 @llvm.nvvm.f16x2.to.e2m1x2.rn.satfinite(<2 x half> %{{.*}})
  // CHECK-NEXT: %{{.*}} = trunc i16 %[[res1]] to i8
  %res1 = nvvm.convert.f16x2.to.f4x2 %srcA : vector<2xf16> -> i8 (f4E2M1FN)
  // CHECK: %[[res2:.*]] = call i16 @llvm.nvvm.f16x2.to.e2m1x2.rn.relu.satfinite(<2 x half> %{{.*}})
  // CHECK-NEXT: %{{.*}} = trunc i16 %[[res2]] to i8
  %res2 = nvvm.convert.f16x2.to.f4x2 %srcA {relu = true} : vector<2xf16> -> i8 (f4E2M1FN)
  llvm.return
}

// -----

// CHECK-LABEL: @convert_bf16x2_to_f4x2
llvm.func @convert_bf16x2_to_f4x2(%srcA : vector<2xbf16>) {
  // CHECK: %[[res1:.*]] = call i16 @llvm.nvvm.bf16x2.to.e2m1x2.rn.satfinite(<2 x bfloat> %{{.*}})
  // CHECK-NEXT: %{{.*}} = trunc i16 %[[res1]] to i8
  %res1 = nvvm.convert.bf16x2.to.f4x2 %srcA : vector<2xbf16> -> i8 (f4E2M1FN)
  // CHECK: %[[res2:.*]] = call i16 @llvm.nvvm.bf16x2.to.e2m1x2.rn.relu.satfinite(<2 x bfloat> %{{.*}})
  // CHECK-NEXT: %{{.*}} = trunc i16 %[[res2]] to i8
  %res2 = nvvm.convert.bf16x2.to.f4x2 %srcA {relu = true} : vector<2xbf16> -> i8 (f4E2M1FN)
  llvm.return
}

// -----

// CHECK-LABEL: @convert_f4x2_to_f16x2
llvm.func @convert_f4x2_to_f16x2(%src : i8) {
  // CHECK: %[[res1:.*]] = zext i8 %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x half> @llvm.nvvm.e2m1x2.to.f16x2.rn(i16 %[[res1]])
  %res1 = nvvm.convert.f4x2.to.f16x2 %src : i8 (f4E2M1FN)-> vector<2xf16>
  // CHECK: %[[res2:.*]] = zext i8 %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x half> @llvm.nvvm.e2m1x2.to.f16x2.rn.relu(i16 %[[res2]])
  %res2 = nvvm.convert.f4x2.to.f16x2 %src {relu = true} : i8 (f4E2M1FN)-> vector<2xf16>
  llvm.return
}

// -----

// CHECK-LABEL: @convert_f4x2_to_bf16x2
llvm.func @convert_f4x2_to_bf16x2(%src : i8, %scale_factor : i16) {
  // CHECK: %[[res1:.*]] = zext i8 %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x bfloat> @llvm.nvvm.e2m1x2.to.bf16x2.rn.scale.n2.ue8m0(i16 %[[res1]], i16 32639)
  %res1 = nvvm.convert.f4x2.to.bf16x2 %src : i8 (f4E2M1FN) -> vector<2xbf16>
  // CHECK: %[[res2:.*]] = zext i8 %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x bfloat> @llvm.nvvm.e2m1x2.to.bf16x2.rn.relu.scale.n2.ue8m0(i16 %[[res2]], i16 32639)
  %res2 = nvvm.convert.f4x2.to.bf16x2 %src {relu = true} : i8 (f4E2M1FN) -> vector<2xbf16>
  // CHECK: %[[res3:.*]] = zext i8 %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x bfloat> @llvm.nvvm.e2m1x2.to.bf16x2.rn.satfinite.scale.n2.ue8m0(i16 %[[res3]], i16 32639)
  %res3 = nvvm.convert.f4x2.to.bf16x2 %src {sat = #nvvm.sat_mode<satfinite>} : i8 (f4E2M1FN) -> vector<2xbf16>
  // CHECK: %[[res4:.*]] = zext i8 %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x bfloat> @llvm.nvvm.e2m1x2.to.bf16x2.rn.relu.satfinite.scale.n2.ue8m0(i16 %[[res4]], i16 32639)
  %res4 = nvvm.convert.f4x2.to.bf16x2 %src {relu = true, sat = #nvvm.sat_mode<satfinite>} : i8 (f4E2M1FN) -> vector<2xbf16>
  // CHECK: %[[res5:.*]] = zext i8 %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x bfloat> @llvm.nvvm.e2m1x2.to.bf16x2.rn.scale.n2.ue8m0(i16 %[[res5]], i16 %{{.*}})
  %res5 = nvvm.convert.f4x2.to.bf16x2 %src, %scale_factor : i8 (f4E2M1FN) -> vector<2xbf16>
  llvm.return
}
