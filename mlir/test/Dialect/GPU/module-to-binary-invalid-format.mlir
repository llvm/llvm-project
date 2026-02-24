// RUN: mlir-opt %s --gpu-module-to-binary="format=invalid-format" --verify-diagnostics

// expected-error @+1 {{Invalid format specified.}}
module attributes {gpu.container_module} {
}
