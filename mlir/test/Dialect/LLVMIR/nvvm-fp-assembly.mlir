// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: llvm.func @fp_modifiers
llvm.func @fp_modifiers(%a: f32, %b: f32, %x: f16, %y: f16, %z: f16) {
  // CHECK: nvvm.addf {{.*}} rnd = <rp> sat = <sat> ftz = true : f32
  %0 = nvvm.addf %a, %b ftz = true sat = <sat> rnd = <rp> : f32

  // CHECK: nvvm.divf {{.*}} ftz = true approx = true : f32
  %1 = nvvm.divf %a, %b approx = true ftz = true : f32

  // CHECK: nvvm.fma {{.*}}, rnd = <rn> relu = true oob = true : f16
  %2 = nvvm.fma %x, %y, %z, rnd = <rn> oob = true relu = true : f16

  // CHECK: nvvm.sqrt {{.*}}, rnd = <rz> ftz = true : f32
  %3 = nvvm.sqrt %a, rnd = <rz> ftz = true : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @conversion_modifiers
llvm.func @conversion_modifiers(%src: f32) {
  // CHECK: nvvm.convert.float.to.tf32 {{.*}} rnd = <rn> sat = <satfinite> relu = true
  %0 = nvvm.convert.float.to.tf32 %src relu = true sat = <satfinite> rnd = <rn>
  llvm.return
}
