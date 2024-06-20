// RUN: mlir-opt %s -enable-arm-streaming="if-compatible-and-scalable if-required-by-ops" -verify-diagnostics

// expected-error@below {{enable-arm-streaming: `if-required-by-ops` and `if-compatible-and-scalable` are mutually exclusive}}
func.func @test() { return }
