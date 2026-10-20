// RUN: %offload-device-info | %fcheck-generic
//
// Just check any device was found and something is printed
//
// CHECK: Num Devices: {{[1-9].*}}
// CHECK: [{{[1-9A-Za-z].*}}]
