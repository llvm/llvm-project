// RUN: mlir-opt %s -enable-arm-streaming="if-contains-scalable-vectors if-required-by-ops" -verify-diagnostics

// expected-error@below {{enable-arm-streaming: `if-required-by-ops` and `if-contains-scalable-vectors` are mutually exclusive}}
func.func @test() { return }
