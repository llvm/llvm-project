// RUN: mlir-opt -allow-unregistered-dialect %s -verify-diagnostics -split-input-file -mlir-print-elementsattrs-with-hex-if-larger=1 | FileCheck %s --check-prefix=HEX
// RUN: mlir-opt -allow-unregistered-dialect %s -verify-diagnostics -split-input-file | FileCheck %s

// HEX: dense<"0x00000000000024400000000000001440"> : tensor<2xf64>
"foo.op"() {dense.attr = dense<[10.0, 5.0]> : tensor<2xf64>} : () -> ()

// CHECK: dense<[1.000000e+01, 5.000000e+00]> : tensor<2xf64>
"foo.op"() {dense.attr = dense<"0x00000000000024400000000000001440"> : tensor<2xf64>} : () -> ()

// CHECK: dense<(1.000000e+01,5.000000e+00)> : tensor<2xcomplex<f64>>
"foo.op"() {dense.attr = dense<"0x0000000000002440000000000000144000000000000024400000000000001440"> : tensor<2xcomplex<f64>>} : () -> ()

// CHECK: dense<[1.000000e+01, 5.000000e+00]> : tensor<2xbf16>
"foo.op"() {dense.attr = dense<"0x2041A040"> : tensor<2xbf16>} : () -> ()

// -----

// expected-error@+1 {{elements hex string should start with '0x'}}
"foo.op"() {dense.attr = dense<"00000000000024400000000000001440"> : tensor<2xf64>} : () -> ()

// -----

// expected-error@+1 {{elements hex string only contains hex digits}}
"foo.op"() {dense.attr = dense<"0x0000000000002440000000000000144X"> : tensor<2xf64>} : () -> ()

// -----

// expected-error@+1 {{elements hex data size is invalid for provided type}}
"foo.op"() {dense.attr = dense<"0x00000000000024400000000000001440"> : tensor<4xf64>} : () -> ()
