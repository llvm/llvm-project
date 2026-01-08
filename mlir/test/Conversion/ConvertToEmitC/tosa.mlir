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
// CHECK:         emitc.verbatim "\0A/* Generalized Indexing Template */
// CHECK-SAME: template <typename T> constexpr T mt_index(T i_last)
// CHECK-SAME:{ return i_last; }\0Atemplate <typename T, typename... Args>
// CHECK-SAME: \0Aconstexpr T mt_index(T idx, T stride, Args... rest)
// CHECK-SAME: {\0A    return (idx * stride) + mt_index(rest...);\0A}\0A"
// CHECK:         emitc.func private @main(%[[ARG0:.*]]: !emitc.ptr<!emitc.array<2xf32>>, %[[ARG1:.*]]: !emitc.ptr<!emitc.array<2xf32>>,
// CHECK-SAME:      %[[ARG2:.*]]: !emitc.ptr<!emitc.array<2xf32>>) attributes {specifiers = ["static"]} {
// CHECK-DAG:       [[VAR_0_:%.+]] = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
// CHECK-DAG:       [[VAR_1_:%.+]] = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
// CHECK-DAG:       [[VAR_2_:%.+]] = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
// CHECK:           for [[I_0_:%.+]] = [[VAR_0_]] to [[VAR_1_]] step [[VAR_2_]]  : !emitc.size_t {
// CHECK-DAG:         [[OPAQUE_INDEX_1:%.+]] = call_opaque "mt_index"([[I_0_]]) : (!emitc.size_t) -> index
// CHECK-DAG:         [[VAR_9_:%.+]] = cast %[[ARG0]] : !emitc.ptr<!emitc.array<2xf32>> to !emitc.ptr<f32>
// CHECK:             [[SUBSCRIPT_1:%.+]] = subscript [[VAR_9_]]{{.}}[[OPAQUE_INDEX_1]]{{.}} : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
// CHECK-DAG:         [[LOAD_MEM_1:%.+]] = load [[SUBSCRIPT_1]] : <f32>
// CHECK-DAG:         [[OPAQUE_INDEX_2:%.+]] = call_opaque "mt_index"([[I_0_]]) : (!emitc.size_t) -> index
// CHECK-DAG:         [[CAST_1:%.+]] = cast %[[ARG1]] : !emitc.ptr<!emitc.array<2xf32>> to !emitc.ptr<f32>
// CHECK:             [[SUBSCRIPT_2:%.+]] = subscript [[CAST_1]]{{.}}[[OPAQUE_INDEX_2]]{{.}} : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
// CHECK:             [[LOAD_MEM_2:%.+]] = load [[SUBSCRIPT_2]] : <f32>
// CHECK-DAG:         [[ADD:%.+]] = add [[LOAD_MEM_1]], [[LOAD_MEM_2]] : (f32, f32) -> f32
// CHECK-DAG:         [[OPAQUE_INDEX_3:%.+]] = call_opaque "mt_index"([[I_0_]]) : (!emitc.size_t) -> index
// CHECK-DAG:         [[CAST_2:%.+]] = cast %[[ARG2]] : !emitc.ptr<!emitc.array<2xf32>> to !emitc.ptr<f32>
// CHECK:             [[SUBSCRIPT_3:%.+]] = subscript [[CAST_2]]{{.}}[[OPAQUE_INDEX_3]]{{.}} : (!emitc.ptr<f32>, index) -> !emitc.lvalue<f32>
// CHECK:             assign [[ADD]] : f32 to [[SUBSCRIPT_3]] : <f32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func private @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
