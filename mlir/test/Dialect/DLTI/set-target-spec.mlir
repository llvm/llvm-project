// REQUIRES: host-supports-nvptx
// RUN: mlir-opt %s --pass-pipeline="builtin.module(gpu.module(dlti-set-target-specs))" | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL:gpu.module @kernel_module1
  // CHECK: dlti.dl_spec = #dlti.dl_spec<
  // CHECK-SAME: !llvm.ptr<6> = dense<32> : vector<4xi64>,
  // CHECK-SAME: i64 = dense<64> : vector<2xi64>,
  // CHECK-SAME: i128 = dense<128> : vector<2xi64>,
  // CHECK-SAME: !llvm.ptr = dense<64> : vector<4xi64>,
  // CHECK-SAME: i1 = dense<8> : vector<2xi64>,
  // CHECK-SAME: i8 = dense<8> : vector<2xi64>,
  // CHECK-SAME: i16 = dense<16> : vector<2xi64>,
  // CHECK-SAME: i32 = dense<32> : vector<2xi64>,
  // CHECK-SAME: f16 = dense<16> : vector<2xi64>,
  // CHECK-SAME: f64 = dense<64> : vector<2xi64>,
  // CHECK-SAME: f128 = dense<128> : vector<2xi64>,
  // CHECK-SAME: "dlti.endianness" = "little">}
  gpu.module @kernel_module1 [#nvvm.target<chip = "sm_70">] {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
