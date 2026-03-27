// RUN: %offload-device-info --json | %fcheck-generic
//
// Just check that a device was found and the JSON output is well-formed.
//
// CHECK: Liboffload Version: {{[0-9]+\.[0-9]+\.[0-9]+}}
// CHECK: Num Devices: {{[1-9].*}}
// CHECK: {"Devices":[{
// CHECK-SAME: "Liboffload Version":"{{[0-9]+\.[0-9]+\.[0-9]+}}"
// CHECK-SAME: "Num Devices":{{[1-9].*}}
