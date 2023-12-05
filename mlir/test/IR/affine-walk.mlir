// RUN: mlir-opt -test-affine-walk -verify-diagnostics %s

// Test affine walk interrupt. A remark should be printed only for the first mod
// expression encountered in post order.

#map = affine_map<(i, j) -> ((i mod 4) mod 2, j)>

"test.check_first_mod"() {"map" = #map} : () -> ()
// expected-remark@-1 {{mod expression}}
