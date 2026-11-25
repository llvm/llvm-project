; Bitcasts between vectors and integers of a non-standard width. Without
; SPV_ALTERA_arbitrary_precision_integers the integer is emitted as the next
; wider integer type, so the bitcast goes through an integer vector padded to
; that width.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-NOEXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers -o - | FileCheck %s --check-prefix=CHECK-EXT

; CHECK-NOEXT-NOT: OpTypeInt 48
; CHECK-NOEXT-NOT: OpTypeInt 24
; CHECK-NOEXT-DAG: %[[#Half:]] = OpTypeFloat 16
; CHECK-NOEXT-DAG: %[[#Vec3Half:]] = OpTypeVector %[[#Half]] 3
; CHECK-NOEXT-DAG: %[[#Int16:]] = OpTypeInt 16 0
; CHECK-NOEXT-DAG: %[[#Vec3Int16:]] = OpTypeVector %[[#Int16]] 3
; CHECK-NOEXT-DAG: %[[#Vec4Int16:]] = OpTypeVector %[[#Int16]] 4
; CHECK-NOEXT-DAG: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-NOEXT-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-NOEXT-DAG: %[[#Vec3Int8:]] = OpTypeVector %[[#Int8]] 3
; CHECK-NOEXT-DAG: %[[#Vec4Int8:]] = OpTypeVector %[[#Int8]] 4
; CHECK-NOEXT-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-NOEXT-DAG: %[[#Mask48:]] = OpConstant %[[#Int64]] 281474976710655
; CHECK-NOEXT-DAG: %[[#Mask24:]] = OpConstant %[[#Int32]] 16777215

; CHECK-EXT-DAG: %[[#Int48:]] = OpTypeInt 48 0
; CHECK-EXT-DAG: %[[#Int24:]] = OpTypeInt 24 0

; CHECK-NOEXT-LABEL: Begin function vec_to_int48
; CHECK-NOEXT: %[[#V:]] = OpFunctionParameter %[[#Vec3Half]]
; CHECK-NOEXT: %[[#VI:]] = OpBitcast %[[#Vec3Int16]] %[[#V]]
; CHECK-NOEXT: %[[#Pad:]] = OpVectorShuffle %[[#Vec4Int16]] %[[#VI]] %[[#VI]] 0 1 2 0xFFFFFFFF
; CHECK-NOEXT: %[[#Wide:]] = OpBitcast %[[#Int64]] %[[#Pad]]
; CHECK-NOEXT: %[[#]] = OpBitwiseAnd %[[#Int64]] %[[#Wide]] %[[#Mask48]]
; CHECK-EXT-LABEL: Begin function vec_to_int48
; CHECK-EXT: OpBitcast %[[#Int48]]
define spir_func i64 @vec_to_int48(<3 x half> %v) {
entry:
  %i = bitcast <3 x half> %v to i48
  %z = zext i48 %i to i64
  ret i64 %z
}

; The bit count reuses the mask of the widened bitcast.
; CHECK-NOEXT-LABEL: Begin function ctpop_of_vec_to_int48
; CHECK-NOEXT: %[[#Wide48:]] = OpBitcast %[[#Int64]] %[[#]]
; CHECK-NOEXT-NEXT: %[[#Masked:]] = OpBitwiseAnd %[[#Int64]] %[[#Wide48]] %[[#Mask48]]
; CHECK-NOEXT-NEXT: %[[#]] = OpBitCount %[[#Int64]] %[[#Masked]]
; CHECK-EXT-LABEL: Begin function ctpop_of_vec_to_int48
; CHECK-EXT: OpBitCount %[[#Int48]]
define spir_func i48 @ctpop_of_vec_to_int48(<3 x half> %v) {
entry:
  %i = bitcast <3 x half> %v to i48
  %c = call i48 @llvm.ctpop.i48(i48 %i)
  ret i48 %c
}

; CHECK-NOEXT-LABEL: Begin function int48_to_vec
; CHECK-NOEXT: %[[#X:]] = OpFunctionParameter %[[#Int64]]
; CHECK-NOEXT: %[[#Masked:]] = OpBitwiseAnd %[[#Int64]] %[[#X]] %[[#Mask48]]
; CHECK-NOEXT: %[[#WideVec:]] = OpBitcast %[[#Vec4Int16]] %[[#Masked]]
; CHECK-NOEXT: %[[#VI2:]] = OpVectorShuffle %[[#Vec3Int16]] %[[#WideVec]] %[[#WideVec]] 0 1 2
; CHECK-NOEXT: %[[#]] = OpBitcast %[[#Vec3Half]] %[[#VI2]]
; CHECK-EXT-LABEL: Begin function int48_to_vec
; CHECK-EXT: OpBitcast %[[#]] %[[#]]
define spir_func <3 x half> @int48_to_vec(i64 %x) {
entry:
  %t = trunc i64 %x to i48
  %v = bitcast i48 %t to <3 x half>
  ret <3 x half> %v
}

; CHECK-NOEXT-LABEL: Begin function vec_to_int24
; CHECK-NOEXT: %[[#V8:]] = OpFunctionParameter %[[#Vec3Int8]]
; CHECK-NOEXT-NEXT: OpLabel
; CHECK-NOEXT-NEXT: %[[#Pad8:]] = OpVectorShuffle %[[#Vec4Int8]] %[[#V8]] %[[#V8]] 0 1 2 0xFFFFFFFF
; CHECK-NOEXT: %[[#Wide32:]] = OpBitcast %[[#Int32]] %[[#Pad8]]
; CHECK-NOEXT: %[[#]] = OpBitwiseAnd %[[#Int32]] %[[#Wide32]] %[[#Mask24]]
; CHECK-EXT-LABEL: Begin function vec_to_int24
; CHECK-EXT: OpBitcast %[[#Int24]]
define spir_func i32 @vec_to_int24(<3 x i8> %v) {
entry:
  %i = bitcast <3 x i8> %v to i24
  %z = zext i24 %i to i32
  ret i32 %z
}

; CHECK-NOEXT-LABEL: Begin function int24_to_vec
; CHECK-NOEXT: %[[#X32:]] = OpFunctionParameter %[[#Int32]]
; CHECK-NOEXT: %[[#Masked32:]] = OpBitwiseAnd %[[#Int32]] %[[#X32]] %[[#Mask24]]
; CHECK-NOEXT: %[[#WideVec8:]] = OpBitcast %[[#Vec4Int8]] %[[#Masked32]]
; CHECK-NOEXT: %[[#Res8:]] = OpVectorShuffle %[[#Vec3Int8]] %[[#WideVec8]] %[[#WideVec8]] 0 1 2
; CHECK-NOEXT-NEXT: OpReturnValue %[[#Res8]]
; CHECK-EXT-LABEL: Begin function int24_to_vec
; CHECK-EXT: OpBitcast %[[#]] %[[#]]
define spir_func <3 x i8> @int24_to_vec(i32 %x) {
entry:
  %t = trunc i32 %x to i24
  %v = bitcast i24 %t to <3 x i8>
  ret <3 x i8> %v
}

declare i48 @llvm.ctpop.i48(i48)
