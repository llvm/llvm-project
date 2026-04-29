// DEFINE: %{pipeline} = "builtin.module(\
// DEFINE:   func.func(\
// DEFINE:     tosa-to-linalg\
// DEFINE:   ),\
// DEFINE:   one-shot-bufferize{\
// DEFINE:     bufferize-function-boundaries\
// DEFINE:     function-boundary-type-conversion=identity-layout-map\
// DEFINE:   },\
// DEFINE:   buffer-results-to-out-params{\
// DEFINE:     hoist-static-allocs=true\
// DEFINE:   },\
// DEFINE:   func.func(\
// DEFINE:     convert-linalg-to-loops\
// DEFINE:   ),\
// DEFINE:   canonicalize,\
// DEFINE:   convert-to-emitc\
// DEFINE: )"

// DEFINE: %{lower_to_emitc} = mlir-opt --pass-pipeline=%{pipeline} %s -o %t
// DEFINE: %{translate} = mlir-translate -mlir-to-cpp %t -o %t.c 
// DEFINE: %{compile} =  %host_cc -include stddef.h -fsyntax-only -Wpedantic -Wall -Werror -Wno-unused %t.c

/// Lower via the pipeline defined above
// RUN: rm -f %t && %{lower_to_emitc} && FileCheck %s --input-file=%t && %{translate} && %{compile}

/// Lower via Transform Dialect
// REDEFINE: %{lower_to_emitc} = mlir-opt %s  \
// REDEFINE: -transform-preload-library='transform-library-paths=%p/td.mlir' \
// REDEFINE: -transform-interpreter -test-transform-dialect-erase-schedule | mlir-opt -convert-to-emitc -o %t
// RUN: rm -f %t && %{lower_to_emitc} && FileCheck %s --input-file=%t && %{translate} && %{compile}

// CHECK-LABEL: @add
// CHECK:     add {{.*}} : (f32, f32) -> f32
func.func private @add(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
