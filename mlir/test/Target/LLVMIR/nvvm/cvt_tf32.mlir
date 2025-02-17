// RUN: mlir-translate -mlir-to-llvmir %s  -split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @convert_float_to_tf32_rna
llvm.func @convert_float_to_tf32_rna(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rna(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rna>}
  llvm.return %res : i32
}

// CHECK-LABEL: @convert_float_to_tf32_rna_sf
llvm.func @convert_float_to_tf32_rna_sf(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rna.satfinite(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rna>, sat = #nvvm.sat_mode<satfinite>}
  llvm.return %res : i32
}

// CHECK-LABEL: @convert_float_to_tf32_rn
llvm.func @convert_float_to_tf32_rn(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rn(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rn>}
  llvm.return %res : i32
}

// CHECK-LABEL: @convert_float_to_tf32_rn_relu
llvm.func @convert_float_to_tf32_rn_relu(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rn.relu(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rn>, relu=true}
  llvm.return %res : i32
}

// CHECK-LABEL: @convert_float_to_tf32_rn_sf
llvm.func @convert_float_to_tf32_rn_sf(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rn.satfinite(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<satfinite>}
  llvm.return %res : i32
}

// CHECK-LABEL: @convert_float_to_tf32_rn_relu_sf
llvm.func @convert_float_to_tf32_rn_relu_sf(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rn.relu.satfinite(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rn>, relu=true, sat = #nvvm.sat_mode<satfinite>}
  llvm.return %res : i32
}

// CHECK-LABEL: @convert_float_to_tf32_rz
llvm.func @convert_float_to_tf32_rz(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rz(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rz>}
  llvm.return %res : i32
}

// CHECK-LABEL: @convert_float_to_tf32_rz_relu
llvm.func @convert_float_to_tf32_rz_relu(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rz.relu(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rz>, relu=true}
  llvm.return %res : i32
}

// CHECK-LABEL: @convert_float_to_tf32_rz_sf
llvm.func @convert_float_to_tf32_rz_sf(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rz.satfinite(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<satfinite>}
  llvm.return %res : i32
}

// CHECK-LABEL: @convert_float_to_tf32_rz_relu_sf
llvm.func @convert_float_to_tf32_rz_relu_sf(%src : f32) -> i32 {
  // CHECK: %{{.*}} = call i32 @llvm.nvvm.f2tf32.rz.relu.satfinite(float %{{.*}})
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rz>, relu=true, sat = #nvvm.sat_mode<satfinite>}
  llvm.return %res : i32
}
