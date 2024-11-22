// DEFINE: %{pipeline} = "builtin.module(\
// DEFINE:   func.func(\
// DEFINE:     tosa-to-linalg\
// DEFINE:   ),\
// DEFINE:   one-shot-bufferize{\
// DEFINE:     bufferize-function-boundaries\
// DEFINE:     function-boundary-type-conversion=identity-layout-map\
// DEFINE:     buffer-alignment=0\
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

// RUN: mlir-opt --pass-pipeline=%{pipeline} %s | FileCheck %s

// -----

//      CHECK: emitc.func @main(%[[ARG0:.*]]: !emitc.array<2xf32>, %[[ARG1:.*]]: !emitc.array<2xf32>, %[[RES:.*]]: !emitc.array<2xf32>) {
//  CHECK-DAG:   %[[C0:.*]] = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
//  CHECK-DAG:   %[[C1:.*]] = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
//  CHECK-DAG:   %[[C2:.*]] = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
//  CHECK-DAG:   %[[I0:.*]] = builtin.unrealized_conversion_cast %[[C0]] : !emitc.size_t to index
//  CHECK-DAG:   %[[I1:.*]] = builtin.unrealized_conversion_cast %[[C1]] : !emitc.size_t to index
//  CHECK-DAG:   %[[I2:.*]] = builtin.unrealized_conversion_cast %[[C2]] : !emitc.size_t to index
// CHECK-NEXT:   for %[[INDEX:.*]] = %[[I0]] to %[[I2]] step %[[I1]] {
// CHECK-NEXT:     %[[INDEX_S:.*]] = builtin.unrealized_conversion_cast %[[INDEX]] : index to !emitc.size_t
// CHECK-NEXT:     %[[V0_LVALUE:.*]] = emitc.subscript %[[ARG0]][%[[INDEX_S]]] : (!emitc.array<2xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:     %[[V0:.*]] = emitc.load %[[V0_LVALUE]] : <f32>
// CHECK-NEXT:     %[[V1_LVALUE:.*]] = emitc.subscript %[[ARG1]][%[[INDEX_S]]] : (!emitc.array<2xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:     %[[V1:.*]] = emitc.load %[[V1_LVALUE]] : <f32>
// CHECK-NEXT:     %[[VADD:.*]] = emitc.add %[[V0]], %[[V1]] : (f32, f32) -> f32
// CHECK-NEXT:     %[[RES_LVALUE:.*]] = emitc.subscript %[[RES]][%[[INDEX_S]]] : (!emitc.array<2xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-NEXT:     emitc.assign %[[VADD]] : f32 to %[[RES_LVALUE]] : <f32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
