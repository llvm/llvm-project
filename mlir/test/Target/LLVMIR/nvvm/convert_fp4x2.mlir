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
