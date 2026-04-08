// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @convert_f32x2_to_fp6x2_e2m3
llvm.func @convert_f32x2_to_fp6x2_e2m3(%srcA : f32, %srcB : f32) {
  //CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e2m3x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.f6x2 %srcA, %srcB : i16 (f6E2M3FN)
  //CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e2m3x2.rn.relu.satfinite(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.f6x2 %srcA, %srcB {relu = true} : i16 (f6E2M3FN)
  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_fp6x2_e3m2
llvm.func @convert_f32x2_to_fp6x2_e3m2(%srcA : f32, %srcB : f32) {
  //CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e3m2x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  %res1 = nvvm.convert.f32x2.to.f6x2 %srcA, %srcB : i16 (f6E3M2FN)
  //CHECK: %{{.*}} = call i16 @llvm.nvvm.ff.to.e3m2x2.rn.relu.satfinite(float %{{.*}}, float %{{.*}})
  %res2 = nvvm.convert.f32x2.to.f6x2 %srcA, %srcB {relu = true} : i16 (f6E3M2FN)
  llvm.return
}

// CHECK-LABEL: @convert_f32x2_to_fp6x2_vector
llvm.func @convert_f32x2_to_fp6x2_vector(%srcA : f32, %srcB : f32) {
  //CHECK: %[[res0:.*]] = call i16 @llvm.nvvm.ff.to.e2m3x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  //CHECK-NEXT: %{{.*}} = bitcast i16 %[[res0]] to <2 x i8>
  %res1 = nvvm.convert.f32x2.to.f6x2 %srcA, %srcB : vector<2xi8> (f6E2M3FN)
  //CHECK: %[[res1:.*]] = call i16 @llvm.nvvm.ff.to.e3m2x2.rn.satfinite(float %{{.*}}, float %{{.*}})
  //CHECK-NEXT: %{{.*}} = bitcast i16 %[[res1]] to <2 x i8>
  %res2 = nvvm.convert.f32x2.to.f6x2 %srcA, %srcB : vector<2xi8> (f6E3M2FN)
  llvm.return
}

// -----

// CHECK-LABEL: @convert_f16x2_to_fp6x2_e2m3
llvm.func @convert_f16x2_to_fp6x2_e2m3(%srcA : vector<2xf16>) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.e2m3x2.rn.satfinite(<2 x half> %{{.*}})
  %res1 = nvvm.convert.f16x2.to.f6x2 %srcA : vector<2xf16> -> i16 (f6E2M3FN)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.e2m3x2.rn.relu.satfinite(<2 x half> %{{.*}})
  %res2 = nvvm.convert.f16x2.to.f6x2 %srcA {relu = true} : vector<2xf16> -> i16 (f6E2M3FN)
  llvm.return
}

// CHECK-LABEL: @convert_f16x2_to_fp6x2_e3m2
llvm.func @convert_f16x2_to_fp6x2_e3m2(%srcA : vector<2xf16>) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.e3m2x2.rn.satfinite(<2 x half> %{{.*}})
  %res1 = nvvm.convert.f16x2.to.f6x2 %srcA : vector<2xf16> -> i16 (f6E3M2FN)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.f16x2.to.e3m2x2.rn.relu.satfinite(<2 x half> %{{.*}})
  %res2 = nvvm.convert.f16x2.to.f6x2 %srcA {relu = true} : vector<2xf16> -> i16 (f6E3M2FN)
  llvm.return
}

// CHECK-LABEL: @convert_f16x2_to_fp6x2_vector
llvm.func @convert_f16x2_to_fp6x2_vector(%srcA : vector<2xf16>) {
  // CHECK: %[[res1:.*]] = call i16 @llvm.nvvm.f16x2.to.e2m3x2.rn.satfinite(<2 x half> %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %[[res1]] to <2 x i8>
  %res1 = nvvm.convert.f16x2.to.f6x2 %srcA : vector<2xf16> -> vector<2xi8> (f6E2M3FN)
  // CHECK: %[[res2:.*]] = call i16 @llvm.nvvm.f16x2.to.e3m2x2.rn.satfinite(<2 x half> %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %[[res2]] to <2 x i8>
  %res2 = nvvm.convert.f16x2.to.f6x2 %srcA : vector<2xf16> -> vector<2xi8> (f6E3M2FN)
  llvm.return
}

// -----

// CHECK-LABEL: @convert_bf16x2_to_fp6x2_e2m3
llvm.func @convert_bf16x2_to_fp6x2_e2m3(%srcA : vector<2xbf16>, %scale_factor : i16) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.e2m3x2.rn.satfinite(<2 x bfloat> %{{.*}})
  %res1 = nvvm.convert.bf16x2.to.f6x2 %srcA : vector<2xbf16> -> i16 (f6E2M3FN)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.e2m3x2.rn.relu.satfinite(<2 x bfloat> %{{.*}})
  %res2 = nvvm.convert.bf16x2.to.f6x2 %srcA {relu = true} : vector<2xbf16> -> i16 (f6E2M3FN)
  llvm.return
}

// CHECK-LABEL: @convert_bf16x2_to_fp6x2_e3m2
llvm.func @convert_bf16x2_to_fp6x2_e3m2(%srcA : vector<2xbf16>, %scale_factor : i16) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.e3m2x2.rn.satfinite(<2 x bfloat> %{{.*}})
  %res1 = nvvm.convert.bf16x2.to.f6x2 %srcA : vector<2xbf16> -> i16 (f6E3M2FN)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.e3m2x2.rn.relu.satfinite(<2 x bfloat> %{{.*}})
  %res2 = nvvm.convert.bf16x2.to.f6x2 %srcA {relu = true} : vector<2xbf16> -> i16 (f6E3M2FN)
  llvm.return
}

// CHECK-LABEL: @convert_bf16x2_to_fp6x2_vector
llvm.func @convert_bf16x2_to_fp6x2_vector(%srcA : vector<2xbf16>, %scale_factor : i16) {
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.e2m3x2.rn.satfinite(<2 x bfloat> %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %{{.*}} to <2 x i8>
  %res1 = nvvm.convert.bf16x2.to.f6x2 %srcA : vector<2xbf16> -> vector<2xi8> (f6E2M3FN)
  // CHECK: %{{.*}} = call i16 @llvm.nvvm.bf16x2.to.e3m2x2.rn.satfinite(<2 x bfloat> %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i16 %{{.*}} to <2 x i8>
  %res2 = nvvm.convert.bf16x2.to.f6x2 %srcA : vector<2xbf16> -> vector<2xi8> (f6E3M2FN)
  llvm.return
}

// -----

// CHECK-LABEL: @convert_f6x2_to_f16x2_e2m3
llvm.func @convert_f6x2_to_f16x2_e2m3(%src : vector<2xi8>) {
  // CHECK: %[[res1:.*]] = bitcast <2 x i8> %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x half> @llvm.nvvm.e2m3x2.to.f16x2.rn(i16 %[[res1]])
  %res1 = nvvm.convert.f6x2.to.f16x2 %src : vector<2xi8> (f6E2M3FN)-> vector<2xf16>
  // CHECK: %[[res2:.*]] = bitcast <2 x i8> %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x half> @llvm.nvvm.e2m3x2.to.f16x2.rn.relu(i16 %[[res2]])
  %res2 = nvvm.convert.f6x2.to.f16x2 %src {relu = true} : vector<2xi8> (f6E2M3FN)-> vector<2xf16>
  llvm.return
}

// CHECK-LABEL: @convert_f6x2_to_f16x2_e3m2
llvm.func @convert_f6x2_to_f16x2_e3m2(%src : vector<2xi8>) {
  // CHECK: %[[res1:.*]] = bitcast <2 x i8> %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x half> @llvm.nvvm.e3m2x2.to.f16x2.rn(i16 %[[res1]])
  %res1 = nvvm.convert.f6x2.to.f16x2 %src : vector<2xi8> (f6E3M2FN)-> vector<2xf16>
  // CHECK: %[[res2:.*]] = bitcast <2 x i8> %{{.*}} to i16
  // CHECK-NEXT: %{{.*}} = call <2 x half> @llvm.nvvm.e3m2x2.to.f16x2.rn.relu(i16 %[[res2]])
  %res2 = nvvm.convert.f6x2.to.f16x2 %src {relu = true} : vector<2xi8> (f6E3M2FN)-> vector<2xf16>
  llvm.return
}
