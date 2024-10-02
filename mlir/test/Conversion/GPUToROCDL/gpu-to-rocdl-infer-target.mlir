// RUN: mlir-opt %s -convert-gpu-to-rocdl -split-input-file --verify-diagnostics | FileCheck --check-prefix=CHECK_TARGET %s

// CHECK_TARGET: @test_module [#rocdl.target<O = 3, chip = "gfx90a">]  attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
gpu.module @test_module [#rocdl.target<O = 3, chip = "gfx90a">] {
  // CHECK_TARGET-LABEL: @kernel_func
  // CHECK_TARGET: attributes
  // CHECK_TARGET: gpu.kernel
  // CHECK_TARGET: rocdl.kernel
  gpu.func @kernel_func() kernel {
    gpu.return
  }
}

// -----

// expected-error@below {{ROCDLTargetAttr is empty on GPU module}}
gpu.module @test_module {
  gpu.func @kernel_func() kernel {
    gpu.return
  }
}

// -----

// expected-error@below {{Invalid chipset name: gfx90a,gfx900}}
gpu.module @test_module [#rocdl.target<O = 3, chip = "gfx90a,gfx900">] {
  gpu.func @kernel_func() kernel {
    gpu.return
  }
}
