// RUN: mlir-opt %s | FileCheck %s

// CHECK: test.would_print_unqualified #test.always_qualified<5> -> !test.always_qualified<7>
%0 = test.would_print_unqualified #test.always_qualified<5> -> !test.always_qualified<7>
