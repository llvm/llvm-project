// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s

include "reg-with-subregs-common.td"

// CHECK-LABEL: static const RegClassWeight RCWeightTable[] = {
// CHECK: {1, 256}, // GPR32
// CHECK: {2, 256}, // GPR_64

// CHECK: {0, 256}, // GPR_64_W0
def GPR_64_W0 : RegisterClass<"", [v2i32], 64, (add GPR64)> {
  let Weight = 0;
}

// CHECK: {1, 256}, // GPR_64_W1
def GPR_64_W1 : RegisterClass<"", [v2i32], 64, (add GPR64)> {
  let Weight = 1;
}

// CHECK: {8, 256}, // GPR_64_W8
def GPR_64_W8 : RegisterClass<"", [v2i32], 64, (add GPR64)> {
  let Weight = 8;
}

// CHECK: {32, 256}, // GPR_1024
