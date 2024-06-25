// RUN: mlir-translate -split-input-file -declare-variables-at-top -mlir-to-cpp -verify-diagnostics %s

// expected-error@+1 {{'func.func' op cannot emit block argument #0 with array type}}
func.func @array_as_block_argument(!emitc.array<4xi8>) {
^bb0(%arg0 : !emitc.array<4xi8>):
  cf.br ^bb1(%arg0 : !emitc.array<4xi8>)
^bb1(%a : !emitc.array<4xi8>):
   return
}
