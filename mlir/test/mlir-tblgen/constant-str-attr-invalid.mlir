// RUN: mlir-opt -verify-diagnostics %s

// Test DefaultValuedAttr<StrAttr, ""> is recognized as "no default value"
test.no_str_value // expected-error {{properties dictionary is missing required attribute: value}}
