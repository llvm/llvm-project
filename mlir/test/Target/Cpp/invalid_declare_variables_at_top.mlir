// RUN: mlir-translate -split-input-file -declare-variables-at-top -mlir-to-cpp -verify-diagnostics %s

// expected-error@+1 {{'func.func' op cannot emit block argument #0 with type '!emitc.array<4xi8>'}}
func.func @array_as_block_argument(!emitc.array<4xi8>) {
^bb0(%arg0 : !emitc.array<4xi8>):
  cf.br ^bb1(%arg0 : !emitc.array<4xi8>)
^bb1(%a : !emitc.array<4xi8>):
  return
}

// -----

// expected-error@+1 {{'emitc.func' op cannot emit block argument #0 with type '!emitc.lvalue<i32>'}}
emitc.func @lvalue_as_block_argument() {
^bb0:
  %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<i32>
  cf.br ^bb1(%0 : !emitc.lvalue<i32>)
^bb1(%a : !emitc.lvalue<i32>):
  emitc.return
}
