; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unkown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unkown-unknown %s -o - -filetype=obj | spirv-val %}
 
; CHECK: OpDecorate %[[#SAT1:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT2:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT3:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT4:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT5:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT6:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT7:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT8:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT9:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT10:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT11:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT12:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT13:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT14:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT15:]] SaturatedConversion
; CHECK: OpDecorate %[[#SAT16:]] SaturatedConversion
 

; CHECK: %[[#SAT1]] = OpConvertFToS %[[#]] %[[#]]
define spir_kernel void @testfunction_float_to_signed_i8(float %input) {
entry:
   %ptr = alloca i8
   %signed_int = call i8 @llvm.fptosi.sat.i8.f32(float %input)
   store i8 %signed_int, ptr %ptr
   ret void

}
declare i8 @llvm.fptosi.sat.i8.f32(float)
 
 
; CHECK: %[[#SAT2]] = OpConvertFToS %[[#]] %[[#]]
define spir_kernel void @testfunction_float_to_signed_i16(float %input) {
entry:
   %ptr = alloca i16
   %signed_int = call i16 @llvm.fptosi.sat.i16.f32(float %input)
   store i16 %signed_int, ptr %ptr
   ret void

}
declare i16 @llvm.fptosi.sat.i16.f32(float)
 
; CHECK: %[[#SAT3]] = OpConvertFToS %[[#]] %[[#]]
define spir_kernel void @testfunction_float_to_signed_i32(float %input) {
entry:
   %ptr = alloca i32
   %signed_int = call i32 @llvm.fptosi.sat.i32.f32(float %input)
   store i32 %signed_int, ptr %ptr
   ret void

}
declare i32 @llvm.fptosi.sat.i32.f32(float)
 

; CHECK: %[[#SAT4]] = OpConvertFToS %[[#]] %[[#]]
define spir_kernel void @testfunction_float_to_signed_i64(float %input) {
entry:
   %ptr = alloca i64
   %signed_int = call i64 @llvm.fptosi.sat.i64.f32(float %input)
   store i64 %signed_int, ptr %ptr
   ret void
}
declare i64 @llvm.fptosi.sat.i64.f32(float)
 

; CHECK: %[[#SAT5]] = OpConvertFToS %[[#]] %[[#]]
define spir_kernel void @testfunction_double_to_signed_i8(double %input) {
entry:
   %ptr = alloca i8
   %signed_int = call i8 @llvm.fptosi.sat.i8.f64(double %input)
   store i8 %signed_int, ptr %ptr
   ret void
}
declare i8 @llvm.fptosi.sat.i8.f64(double)
 

; CHECK: %[[#SAT6]] = OpConvertFToS %[[#]] %[[#]]
define spir_kernel void @testfunction_double_to_signed_i16(double %input) {
entry:
   %ptr = alloca i16
   %signed_int = call i16 @llvm.fptosi.sat.i16.f64(double %input)
   store i16 %signed_int, ptr %ptr
   ret void
}
declare i16 @llvm.fptosi.sat.i16.f64(double)
 
 
; CHECK: %[[#SAT7]] = OpConvertFToS %[[#]] %[[#]]
define spir_kernel void @testfunction_double_to_signed_i32(double %input) {
entry:
   %ptr = alloca i32
   %signed_int = call i32 @llvm.fptosi.sat.i32.f64(double %input)
   store i32 %signed_int, ptr %ptr
   ret void
}
declare i32 @llvm.fptosi.sat.i32.f64(double)
 
 
; CHECK: %[[#SAT8]] = OpConvertFToS %[[#]] %[[#]]
define spir_kernel void @testfunction_double_to_signed_i64(double %input) {
entry:
   %ptr = alloca i64
   %signed_int = call i64 @llvm.fptosi.sat.i64.f64(double %input)
   store i64 %signed_int, ptr %ptr
   ret void
}
declare i64 @llvm.fptosi.sat.i64.f64(double)
 
; CHECK: %[[#SAT9]] = OpConvertFToU %[[#]] %[[#]]
define spir_kernel void @testfunction_float_to_unsigned_i8(float %input) {
entry:
   %ptr = alloca i8
   %unsigned_int = call i8 @llvm.fptoui.sat.i8.f32(float %input)
   store i8 %unsigned_int, ptr %ptr
   ret void
}
declare i8 @llvm.fptoui.sat.i8.f32(float)
 

; CHECK: %[[#SAT10]] = OpConvertFToU %[[#]] %[[#]]
define spir_kernel void @testfunction_float_to_unsigned_i16(float %input) {
entry:
   %ptr = alloca i16
   %unsigned_int = call i16 @llvm.fptoui.sat.i16.f32(float %input)
   store i16 %unsigned_int, ptr %ptr
   ret void
}
declare i16 @llvm.fptoui.sat.i16.f32(float)


; CHECK: %[[#SAT11]] = OpConvertFToU %[[#]] %[[#]]
define spir_kernel void @testfunction_float_to_unsigned_i32(float %input) {
entry:
   %ptr = alloca i32
   %unsigned_int = call i32 @llvm.fptoui.sat.i32.f32(float %input)
   store i32 %unsigned_int, ptr %ptr
   ret void
}
declare i32 @llvm.fptoui.sat.i32.f32(float)
 

; CHECK: %[[#SAT12]] = OpConvertFToU %[[#]] %[[#]]
define spir_kernel void @testfunction_float_to_unsigned_i64(float %input) {
entry:
   %ptr = alloca i64
   %unsigned_int = call i64 @llvm.fptoui.sat.i64.f32(float %input)
   store i64 %unsigned_int, ptr %ptr
   ret void
}
declare i64 @llvm.fptoui.sat.i64.f32(float)
 

; CHECK: %[[#SAT13]] = OpConvertFToU %[[#]] %[[#]]
define spir_kernel void @testfunction_double_to_unsigned_i8(double %input) {
entry:
   %ptr = alloca i8
   %unsigned_int = call i8 @llvm.fptoui.sat.i8.f64(double %input)
   store i8 %unsigned_int, ptr %ptr
   ret void
}
declare i8 @llvm.fptoui.sat.i8.f64(double)
 

; CHECK: %[[#SAT14]] = OpConvertFToU %[[#]] %[[#]]
define spir_kernel void @testfunction_double_to_unsigned_i16(double %input) {
entry:
   %ptr = alloca i16
   %unsigned_int = call i16 @llvm.fptoui.sat.i16.f64(double %input)
   store i16 %unsigned_int, ptr %ptr
   ret void
}
declare i16 @llvm.fptoui.sat.i16.f64(double)
 

; CHECK: %[[#SAT15]] = OpConvertFToU %[[#]] %[[#]]
define spir_kernel void @testfunction_double_to_unsigned_i32(double %input) {
entry:
   %ptr = alloca i32
   %unsigned_int = call i32 @llvm.fptoui.sat.i32.f64(double %input)
   store i32 %unsigned_int, ptr %ptr
   ret void
}
declare i32 @llvm.fptoui.sat.i32.f64(double)
 

; CHECK: %[[#SAT16]] = OpConvertFToU %[[#]] %[[#]]
define spir_kernel void @testfunction_double_to_unsigned_i64(double %input) {
entry:
   %ptr = alloca i64
   %unsigned_int = call i64 @llvm.fptoui.sat.i64.f64(double %input)
   store i64 %unsigned_int, ptr %ptr
   ret void
}
declare i64 @llvm.fptoui.sat.i64.f64(double)
