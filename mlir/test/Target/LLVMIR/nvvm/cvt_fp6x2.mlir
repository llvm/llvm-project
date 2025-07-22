// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @convert_f32x2_to_fp6x2_packed
llvm.func @convert_f32x2_to_fp6x2_packed(%srcA : f32, %srcB : f32) {
  //CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e2m3x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.cvt.f32x2.to.f6x2 <e2m3> %srcA, %srcB : i16
  //CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e3m2x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.cvt.f32x2.to.f6x2 <e3m2> %srcA, %srcB : i16
  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_fp6x2_vector
llvm.func @convert_f32x2_to_fp6x2_vector(%srcA : f32, %srcB : f32) {
  //CHECK: %[[res0:.*]] = call i16 @llvm.nvvm.ff.to.e2m3x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  //CHECK-NEXT: %{{.*}} = bitcast i16 %[[res0]] to <2 x i8>
  %res1 = nvvm.cvt.f32x2.to.f6x2 <e2m3> %srcA, %srcB : vector<2xi8>
  //CHECK: %[[res1:.*]] = call i16 @llvm.nvvm.ff.to.e3m2x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  //CHECK-NEXT: %{{.*}} = bitcast i16 %[[res1]] to <2 x i8>
  %res2 = nvvm.cvt.f32x2.to.f6x2 <e3m2> %srcA, %srcB : vector<2xi8>
  llvm.return
}
