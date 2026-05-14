// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @convert_f32x2_to_s2f6x2(%srcA : f32, %srcB : f32) -> i16 {
  // CHECK-LABEL: define i16 @convert_f32x2_to_s2f6x2(float %0, float %1) {
  // CHECK-NEXT: %3 = call i16 @llvm.nvvm.ff.to.s2f6x2.rn.satfinite.scale.n2.ue8m0(float %0, float %1, i16 32639)
  // CHECK-NEXT: %4 = call i16 @llvm.nvvm.ff.to.s2f6x2.rn.relu.satfinite.scale.n2.ue8m0(float %0, float %1, i16 32639)
  // CHECK-NEXT: %5 = or i16 %3, %4
  // CHECK-NEXT: ret i16 %5
  // CHECK-NEXT: }
  %res1 = nvvm.convert.f32x2.to.s2f6x2 %srcA, %srcB : i16
  %res2 = nvvm.convert.f32x2.to.s2f6x2 %srcA, %srcB {relu = true} : i16

  // Combine results to avoid dead code elimination
  %final_result = llvm.or %res1, %res2 : i16
  llvm.return %final_result : i16
}

llvm.func @convert_f32x2_to_s2f6x2_scale(%srcA : f32, %srcB : f32, %scale : i16) -> i16 {
  // CHECK-LABEL: define i16 @convert_f32x2_to_s2f6x2_scale(float %0, float %1, i16 %2) {
  // CHECK-NEXT: %4 = call i16 @llvm.nvvm.ff.to.s2f6x2.rn.satfinite.scale.n2.ue8m0(float %0, float %1, i16 %2)
  // CHECK-NEXT: %5 = call i16 @llvm.nvvm.ff.to.s2f6x2.rn.relu.satfinite.scale.n2.ue8m0(float %0, float %1, i16 %2)
  // CHECK-NEXT: %6 = or i16 %4, %5
  // CHECK-NEXT: ret i16 %6
  // CHECK-NEXT: }
  %res1 = nvvm.convert.f32x2.to.s2f6x2 %srcA, %srcB, %scale : i16
  %res2 = nvvm.convert.f32x2.to.s2f6x2 %srcA, %srcB, %scale {relu = true} : i16

  // Combine results to avoid dead code elimination
  %final_result = llvm.or %res1, %res2 : i16
  llvm.return %final_result : i16
}

llvm.func @convert_f32x2_to_s2f6x2_vector(%srcA : f32, %srcB : f32) -> vector<2xi8> {
  // CHECK-LABEL: define <2 x i8> @convert_f32x2_to_s2f6x2_vector(float %0, float %1) {
  // CHECK-NEXT: %3 = call i16 @llvm.nvvm.ff.to.s2f6x2.rn.satfinite.scale.n2.ue8m0(float %0, float %1, i16 32639)
  // CHECK-NEXT: %4 = bitcast i16 %3 to <2 x i8>
  // CHECK-NEXT: ret <2 x i8> %4
  // CHECK-NEXT: }
  %res1 = nvvm.convert.f32x2.to.s2f6x2 %srcA, %srcB : vector<2xi8>
  llvm.return %res1 : vector<2xi8>
}

llvm.func @convert_f32x2_to_s2f6x2_vector_scale(%srcA : f32, %srcB : f32, %scale : i16) -> vector<2xi8> {
  // CHECK-LABEL: define <2 x i8> @convert_f32x2_to_s2f6x2_vector_scale(float %0, float %1, i16 %2) {
  // CHECK-NEXT: %4 = call i16 @llvm.nvvm.ff.to.s2f6x2.rn.satfinite.scale.n2.ue8m0(float %0, float %1, i16 %2)
  // CHECK-NEXT: %5 = bitcast i16 %4 to <2 x i8>
  // CHECK-NEXT: ret <2 x i8> %5
  // CHECK-NEXT: }
  %res1 = nvvm.convert.f32x2.to.s2f6x2 %srcA, %srcB, %scale : vector<2xi8>
  llvm.return %res1 : vector<2xi8>
}

llvm.func @convert_bf16x2_to_s2f6x2(%srcA : vector<2xbf16>) -> i16 {
  // CHECK-LABEL: define i16 @convert_bf16x2_to_s2f6x2(<2 x bfloat> %0) {
  // CHECK-NEXT: %2 = call i16 @llvm.nvvm.bf16x2.to.s2f6x2.rn.satfinite.scale.n2.ue8m0(<2 x bfloat> %0, i16 32639)
  // CHECK-NEXT: %3 = call i16 @llvm.nvvm.bf16x2.to.s2f6x2.rn.relu.satfinite.scale.n2.ue8m0(<2 x bfloat> %0, i16 32639)
  // CHECK-NEXT: %4 = or i16 %2, %3
  // CHECK-NEXT: ret i16 %4
  // CHECK-NEXT: }
  %res1 = nvvm.convert.bf16x2.to.s2f6x2 %srcA : vector<2xbf16> -> i16
  %res2 = nvvm.convert.bf16x2.to.s2f6x2 %srcA {relu = true} : vector<2xbf16> -> i16

  // Combine results to avoid dead code elimination
  %final_result = llvm.or %res1, %res2 : i16
  llvm.return %final_result : i16
}

llvm.func @convert_bf16x2_to_s2f6x2_scale(%srcA : vector<2xbf16>, %scale : i16) -> i16 {
  // CHECK-LABEL: define i16 @convert_bf16x2_to_s2f6x2_scale(<2 x bfloat> %0, i16 %1) {
  // CHECK-NEXT: %3 = call i16 @llvm.nvvm.bf16x2.to.s2f6x2.rn.satfinite.scale.n2.ue8m0(<2 x bfloat> %0, i16 %1)
  // CHECK-NEXT: %4 = call i16 @llvm.nvvm.bf16x2.to.s2f6x2.rn.relu.satfinite.scale.n2.ue8m0(<2 x bfloat> %0, i16 %1)
  // CHECK-NEXT: %5 = or i16 %3, %4
  // CHECK-NEXT: ret i16 %5
  // CHECK-NEXT: }
  %res1 = nvvm.convert.bf16x2.to.s2f6x2 %srcA, %scale : vector<2xbf16> -> i16
  %res2 = nvvm.convert.bf16x2.to.s2f6x2 %srcA, %scale {relu = true} : vector<2xbf16> -> i16

  // Combine results to avoid dead code elimination
  %final_result = llvm.or %res1, %res2 : i16
  llvm.return %final_result : i16
}

llvm.func @convert_bf16x2_to_s2f6x2_vector(%srcA : vector<2xbf16>) -> vector<2xi8> {
  // CHECK-LABEL: define <2 x i8> @convert_bf16x2_to_s2f6x2_vector(<2 x bfloat> %0) {
  // CHECK-NEXT: %2 = call i16 @llvm.nvvm.bf16x2.to.s2f6x2.rn.satfinite.scale.n2.ue8m0(<2 x bfloat> %0, i16 32639)
  // CHECK-NEXT: %3 = bitcast i16 %2 to <2 x i8>
  // CHECK-NEXT: ret <2 x i8> %3
  // CHECK-NEXT: }
  %res1 = nvvm.convert.bf16x2.to.s2f6x2 %srcA : vector<2xbf16> -> vector<2xi8>
  llvm.return %res1 : vector<2xi8>
}

llvm.func @convert_bf16x2_to_s2f6x2_vector_scale(%srcA : vector<2xbf16>, %scale : i16) -> vector<2xi8> {
  // CHECK-LABEL: define <2 x i8> @convert_bf16x2_to_s2f6x2_vector_scale(<2 x bfloat> %0, i16 %1) {
  // CHECK-NEXT: %3 = call i16 @llvm.nvvm.bf16x2.to.s2f6x2.rn.satfinite.scale.n2.ue8m0(<2 x bfloat> %0, i16 %1)
  // CHECK-NEXT: %4 = bitcast i16 %3 to <2 x i8>
  // CHECK-NEXT: ret <2 x i8> %4
  // CHECK-NEXT: }
  %res1 = nvvm.convert.bf16x2.to.s2f6x2 %srcA, %scale : vector<2xbf16> -> vector<2xi8>
  llvm.return %res1 : vector<2xi8>
}

// 1. no relu, no scale, no satfinite
llvm.func @convert_s2f6x2_to_bf16x2(%src : vector<2xi8>) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @convert_s2f6x2_to_bf16x2(<2 x i8> %0) {
  // CHECK-NEXT: %2 = bitcast <2 x i8> %0 to i16
  // CHECK-NEXT: %3 = call <2 x bfloat> @llvm.nvvm.s2f6x2.to.bf16x2.rn.scale.n2.ue8m0(i16 %2, i16 32639)
  // CHECK-NEXT: ret <2 x bfloat> %3
  // CHECK-NEXT: }
  %res = nvvm.convert.s2f6x2.to.bf16x2 %src : vector<2xi8> -> vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// 2. relu, no scale, no satfinite
llvm.func @convert_s2f6x2_to_bf16x2_relu(%src : vector<2xi8>) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @convert_s2f6x2_to_bf16x2_relu(<2 x i8> %0) {
  // CHECK-NEXT: %2 = bitcast <2 x i8> %0 to i16
  // CHECK-NEXT: %3 = call <2 x bfloat> @llvm.nvvm.s2f6x2.to.bf16x2.rn.relu.scale.n2.ue8m0(i16 %2, i16 32639)
  // CHECK-NEXT: ret <2 x bfloat> %3
  // CHECK-NEXT: }
  %res = nvvm.convert.s2f6x2.to.bf16x2 %src {relu = true} : vector<2xi8> -> vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// 3. no relu, with scale, no satfinite
llvm.func @convert_s2f6x2_to_bf16x2_scale(%src : vector<2xi8>, %scale : i16) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @convert_s2f6x2_to_bf16x2_scale(<2 x i8> %0, i16 %1) {
  // CHECK-NEXT: %3 = bitcast <2 x i8> %0 to i16
  // CHECK-NEXT: %4 = call <2 x bfloat> @llvm.nvvm.s2f6x2.to.bf16x2.rn.scale.n2.ue8m0(i16 %3, i16 %1)
  // CHECK-NEXT: ret <2 x bfloat> %4
  // CHECK-NEXT: }
  %res = nvvm.convert.s2f6x2.to.bf16x2 %src, %scale : vector<2xi8> -> vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// 4. relu, with scale, no satfinite
llvm.func @convert_s2f6x2_to_bf16x2_scale_relu(%src : vector<2xi8>, %scale : i16) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @convert_s2f6x2_to_bf16x2_scale_relu(<2 x i8> %0, i16 %1) {
  // CHECK-NEXT: %3 = bitcast <2 x i8> %0 to i16
  // CHECK-NEXT: %4 = call <2 x bfloat> @llvm.nvvm.s2f6x2.to.bf16x2.rn.relu.scale.n2.ue8m0(i16 %3, i16 %1)
  // CHECK-NEXT: ret <2 x bfloat> %4
  // CHECK-NEXT: }
  %res = nvvm.convert.s2f6x2.to.bf16x2 %src, %scale {relu = true} : vector<2xi8> -> vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// 5. no relu, no scale, satfinite
llvm.func @convert_s2f6x2_to_bf16x2_satfinite(%src : vector<2xi8>) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @convert_s2f6x2_to_bf16x2_satfinite(<2 x i8> %0) {
  // CHECK-NEXT: %2 = bitcast <2 x i8> %0 to i16
  // CHECK-NEXT: %3 = call <2 x bfloat> @llvm.nvvm.s2f6x2.to.bf16x2.rn.satfinite.scale.n2.ue8m0(i16 %2, i16 32639)
  // CHECK-NEXT: ret <2 x bfloat> %3
  // CHECK-NEXT: }
  %res = nvvm.convert.s2f6x2.to.bf16x2 %src {sat = #nvvm.sat_mode<satfinite>} : vector<2xi8> -> vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// 6. relu, no scale, satfinite
llvm.func @convert_s2f6x2_to_bf16x2_relu_satfinite(%src : vector<2xi8>) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @convert_s2f6x2_to_bf16x2_relu_satfinite(<2 x i8> %0) {
  // CHECK-NEXT: %2 = bitcast <2 x i8> %0 to i16
  // CHECK-NEXT: %3 = call <2 x bfloat> @llvm.nvvm.s2f6x2.to.bf16x2.rn.relu.satfinite.scale.n2.ue8m0(i16 %2, i16 32639)
  // CHECK-NEXT: ret <2 x bfloat> %3
  // CHECK-NEXT: }
  %res = nvvm.convert.s2f6x2.to.bf16x2 %src {relu = true, sat = #nvvm.sat_mode<satfinite>} : vector<2xi8> -> vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// 7. no relu, with scale, satfinite
llvm.func @convert_s2f6x2_to_bf16x2_scale_satfinite(%src : vector<2xi8>, %scale : i16) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @convert_s2f6x2_to_bf16x2_scale_satfinite(<2 x i8> %0, i16 %1) {
  // CHECK-NEXT: %3 = bitcast <2 x i8> %0 to i16
  // CHECK-NEXT: %4 = call <2 x bfloat> @llvm.nvvm.s2f6x2.to.bf16x2.rn.satfinite.scale.n2.ue8m0(i16 %3, i16 %1)
  // CHECK-NEXT: ret <2 x bfloat> %4
  // CHECK-NEXT: }
  %res = nvvm.convert.s2f6x2.to.bf16x2 %src, %scale {sat = #nvvm.sat_mode<satfinite>} : vector<2xi8> -> vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}

// 8. relu, with scale, satfinite
llvm.func @convert_s2f6x2_to_bf16x2_scale_relu_satfinite(%src : vector<2xi8>, %scale : i16) -> vector<2xbf16> {
  // CHECK-LABEL: define <2 x bfloat> @convert_s2f6x2_to_bf16x2_scale_relu_satfinite(<2 x i8> %0, i16 %1) {
  // CHECK-NEXT: %3 = bitcast <2 x i8> %0 to i16
  // CHECK-NEXT: %4 = call <2 x bfloat> @llvm.nvvm.s2f6x2.to.bf16x2.rn.relu.satfinite.scale.n2.ue8m0(i16 %3, i16 %1)
  // CHECK-NEXT: ret <2 x bfloat> %4
  // CHECK-NEXT: }
  %res = nvvm.convert.s2f6x2.to.bf16x2 %src, %scale {relu = true, sat = #nvvm.sat_mode<satfinite>} : vector<2xi8> -> vector<2xbf16>
  llvm.return %res : vector<2xbf16>
}
