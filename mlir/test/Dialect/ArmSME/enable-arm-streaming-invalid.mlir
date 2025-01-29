// RUN: mlir-opt %s -enable-arm-streaming="if-scalable-and-supported if-required-by-ops" -verify-diagnostics

// expected-error@below {{enable-arm-streaming: `if-required-by-ops` and `if-scalable-and-supported` are mutually exclusive}}
func.func @test() { return }
