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
