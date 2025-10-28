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

// CHECK: emitc.func private @main(%[[ARG0:.*]]: !emitc.ptr<!emitc.array<2xf32>>, %[[ARG1:.*]]: !emitc.ptr<!emitc.array<2xf32>>, %[[RES:.*]]: !emitc.ptr<!emitc.array<2xf32>>)
// CHECK-DAG:       [[VAR_0_:%.+]] = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK-DAG:       [[VAR_1_:%.+]] = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
// CHECK-DAG:       [[VAR_2_:%.+]] = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
// CHECK:           for [[I_0_:%.+]] = [[VAR_0_]] to [[VAR_1_]] step [[VAR_2_]]  : !emitc.size_t {
// CHECK:             [[VAR_3_:%.+]] = apply "*"(%[[ARG0]]) : (!emitc.ptr<!emitc.array<2xf32>>) -> !emitc.array<2xf32>
// CHECK:             [[VAR_4_:%.+]] = subscript [[VAR_3_]]{{.}}[[I_0_]]{{.}} : (!emitc.array<2xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK-DAG:         [[LOAD_VAR_4_MEM_:%.+]] = load [[VAR_4_]] : <f32>
// CHECK-DAG:         [[VAR_6_:%.+]] = apply "*"(%[[ARG1]]) : (!emitc.ptr<!emitc.array<2xf32>>) -> !emitc.array<2xf32>
// CHECK:             [[VAR_7_:%.+]] = subscript [[VAR_6_]]{{.}}[[I_0_]]{{.}} : (!emitc.array<2xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK:             [[LOAD_VAR_7_MEM_:%.+]] = load [[VAR_7_]] : <f32>
// CHECK-DAG:         [[VAR_9_:%.+]] = add [[LOAD_VAR_4_MEM_]], [[LOAD_VAR_7_MEM_]] : (f32, f32) -> f32
// CHECK-DAG:         [[VAR_10_:%.+]] = apply "*"(%[[RES]]) : (!emitc.ptr<!emitc.array<2xf32>>) -> !emitc.array<2xf32>
// CHECK:             [[VAR_11_:%.+]] = subscript [[VAR_10_]]{{.}}[[I_0_]]{{.}} : (!emitc.array<2xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
// CHECK:             assign [[VAR_9_]] : f32 to [[VAR_11_]] : <f32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func private @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
