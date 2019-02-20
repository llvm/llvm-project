; RUN: llc -wasm-enable-unimplemented-simd -mattr=+sign-ext,+simd128 -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

declare i32 @i32()
declare i64 @i64()
declare float @f32()
declare double @f64()
declare <16 x i8> @v16i8()
declare <8 x i16> @v8i16()
declare <4 x i32> @v4i32()
declare <2 x i64> @v2i64()
declare <4 x float> @v4f32()
declare <2 x double> @v2f64()
declare { i32, i32, i32 } @structret()

define void @f1() {
entry:
  %tmp1 = call i32 @i32()
  %tmp2 = call i64 @i64()
  %tmp3 = call float @f32()
  %tmp4 = call double @f64()
  %tmp5 = call <16 x i8> @v16i8()
  %tmp6 = call <8 x i16> @v8i16()
  %tmp7 = call <4 x i32> @v4i32()
  %tmp8 = call <2 x i64> @v2i64()
  %tmp9 = call <4 x float> @v4f32()
  %tmp10 = call <2 x double> @v2f64()
  %tmp11 = call {i32, i32, i32} @structret()
  ret void
}

define void @vararg(i32, i32, ...) {
  ret void
}

; CHECK-LABEL: - Type: TYPE
; CHECK-NEXT:    Signatures:
; CHECK-NEXT:       - Index: 0
; CHECK-NEXT:         ReturnType: NORESULT
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:       - Index: 1
; CHECK-NEXT:         ReturnType: I32
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:       - Index: 2
; CHECK-NEXT:         ReturnType: I64
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:       - Index: 3
; CHECK-NEXT:         ReturnType: F32
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:       - Index: 4
; CHECK-NEXT:         ReturnType: F64
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:       - Index: 5
; CHECK-NEXT:         ReturnType: V128
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:       - Index: 6
; CHECK-NEXT:         ReturnType: NORESULT
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:       - Index: 7
; CHECK-NEXT:         ReturnType: NORESULT
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:           - I32
; CHECK-NEXT:           - I32
; should be no additional types
; CHECK-NOT: ReturnType
