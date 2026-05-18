// RUN: mlir-opt -convert-to-emitc -verify-diagnostics %s

/// arith.constant -> emitc.constant conversion cannot make a valid !emitc.ptr<...> / !emitc.array<1xi64>
/// from a dense memref attribute, as it changes the result type, not the attribute.
func.func private @constant_rank0() -> memref<i64> {
  // expected-error@+1 {{'emitc.constant' op requires attribute to either be an #emitc.opaque attribute or it's type ('memref<i64>') to match the op's result type ('!emitc.ptr<i64>')}}
  %0 = arith.constant dense<-1> : memref<i64>
  return %0 : memref<i64>
}

func.func private @constant_rank1() -> memref<1xi64> {
  // expected-error@+1 {{'emitc.constant' op requires attribute to either be an #emitc.opaque attribute or it's type ('memref<1xi64>') to match the op's result type ('!emitc.array<1xi64>')}}
  %0 = arith.constant dense<[-1]> : memref<1xi64>
  return %0 : memref<1xi64>
}

// expected-error@+1 {{'emitc.func' op cannot return array type}}
func.func private @multiple_elements(%arg0: memref<2xi64>) -> memref<2xi64> {
  return %arg0 : memref<2xi64>
}

// expected-error@+1 {{'emitc.func' op cannot return array type}}
func.func @public_function(%arg0: memref<1xi64>) -> memref<1xi64> {
  return %arg0 : memref<1xi64>
}

// expected-error@+1 {{'emitc.func' op cannot return array type}}
func.func private @callee(%arg0: memref<1xi64>) -> memref<1xi64> {
  return %arg0 : memref<1xi64>
}

func.func private @caller(%arg0: memref<1xi64>) -> memref<1xi64> {
  // expected-error@+1 {{'emitc.call' op result #0 must be variadic of type supported by EmitC, but got 'memref<1xi64>'}}
  %0 = call @callee(%arg0) : (memref<1xi64>) -> memref<1xi64>
  return %0 : memref<1xi64>
}
