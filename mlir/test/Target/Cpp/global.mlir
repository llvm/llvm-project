// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s

emitc.global extern @decl : i8
// CHECK: extern int8_t decl;

emitc.global @uninit : i32
// CHECK: int32_t uninit;

emitc.global @myglobal_int : i32 = 4
// CHECK: int32_t myglobal_int = 4;

emitc.global @myglobal : !emitc.array<2xf32> = dense<4.000000e+00>
// CHECK: float myglobal[2] = {4.000000000e+00f, 4.000000000e+00f};

emitc.global const @myconstant : !emitc.array<2xi16> = dense<2>
// CHECK: const int16_t myconstant[2] = {2, 2};

emitc.global extern const @extern_constant : !emitc.array<2xi16>
// CHECK: extern const int16_t extern_constant[2];

emitc.global static @static_var : f32
// CHECK: static float static_var;

emitc.global static @static_const : f32 = 3.0
// CHECK: static float static_const = 3.000000000e+00f;

emitc.global @opaque_init : !emitc.opaque<"char"> = #emitc.opaque<"CHAR_MIN">
// CHECK: char opaque_init = CHAR_MIN;

func.func @use_global(%i: index) -> f32 {
  %0 = emitc.get_global @myglobal : !emitc.array<2xf32>
  %1 = emitc.subscript %0[%i] : (!emitc.array<2xf32>, index) -> f32
  return %1 : f32
  // CHECK-LABEL: use_global
  // CHECK-SAME: (size_t [[V1:.*]])
  // CHECK:   return myglobal[[[V1]]];
}
