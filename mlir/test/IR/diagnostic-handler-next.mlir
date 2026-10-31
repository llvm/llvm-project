// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(test-diagnostic-next))" -verify-diagnostics 

// expected-remark @+3 {{test1}}
// expected-remark @+2 {{test3}}
// expected-remark @+1 {{test2}}
func.func @test() {
  return
}
