// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// A missing `DefaultValuedAttr` of an `AttrDef` is set to its default, and the
// default is elided when printing.

// CHECK: test.with_default_valued_attr_def{{$}}
// GENERIC: "test.with_default_valued_attr_def"() <{attr = #test.default_valued_attr_def<0>}>
test.with_default_valued_attr_def

// CHECK: test.with_default_valued_attr_def{{$}}
// GENERIC: "test.with_default_valued_attr_def"() <{attr = #test.default_valued_attr_def<0>}>
test.with_default_valued_attr_def attr #test.default_valued_attr_def<0>

// CHECK: test.with_default_valued_attr_def attr <1>{{$}}
// GENERIC: "test.with_default_valued_attr_def"() <{attr = #test.default_valued_attr_def<1>}>
test.with_default_valued_attr_def attr #test.default_valued_attr_def<1>
