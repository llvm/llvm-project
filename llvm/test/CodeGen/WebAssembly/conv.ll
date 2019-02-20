; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+nontrapping-fptoint | FileCheck %s

; Test that basic conversion operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: i32_wrap_i64:
; CHECK-NEXT: .functype i32_wrap_i64 (i64) -> (i32){{$}}
; CHECK-NEXT: i32.wrap/i64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @i32_wrap_i64(i64 %x) {
  %a = trunc i64 %x to i32
  ret i32 %a
}

; CHECK-LABEL: i64_extend_s_i32:
; CHECK-NEXT: .functype i64_extend_s_i32 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.extend_s/i32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_extend_s_i32(i32 %x) {
  %a = sext i32 %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_extend_u_i32:
; CHECK-NEXT: .functype i64_extend_u_i32 (i32) -> (i64){{$}}
; CHECK-NEXT: i64.extend_u/i32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_extend_u_i32(i32 %x) {
  %a = zext i32 %x to i64
  ret i64 %a
}

; CHECK-LABEL: i32_trunc_s_f32:
; CHECK-NEXT: .functype i32_trunc_s_f32 (f32) -> (i32){{$}}
; CHECK-NEXT: i32.trunc_s:sat/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @i32_trunc_s_f32(float %x) {
  %a = fptosi float %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_sat_s_f32:
; CHECK-NEXT: .functype i32_trunc_sat_s_f32 (f32) -> (i32){{$}}
; CHECK-NEXT: i32.trunc_s:sat/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
declare i32 @llvm.wasm.trunc.saturate.signed.i32.f32(float)
define i32 @i32_trunc_sat_s_f32(float %x) {
  %a = call i32 @llvm.wasm.trunc.saturate.signed.i32.f32(float %x)
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_u_f32:
; CHECK-NEXT: .functype i32_trunc_u_f32 (f32) -> (i32){{$}}
; CHECK-NEXT: i32.trunc_u:sat/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @i32_trunc_u_f32(float %x) {
  %a = fptoui float %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_sat_u_f32:
; CHECK-NEXT: .functype i32_trunc_sat_u_f32 (f32) -> (i32){{$}}
; CHECK-NEXT: i32.trunc_u:sat/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
declare i32 @llvm.wasm.trunc.saturate.unsigned.i32.f32(float)
define i32 @i32_trunc_sat_u_f32(float %x) {
  %a = call i32 @llvm.wasm.trunc.saturate.unsigned.i32.f32(float %x)
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_s_f64:
; CHECK-NEXT: .functype i32_trunc_s_f64 (f64) -> (i32){{$}}
; CHECK-NEXT: i32.trunc_s:sat/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @i32_trunc_s_f64(double %x) {
  %a = fptosi double %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_sat_s_f64:
; CHECK-NEXT: .functype i32_trunc_sat_s_f64 (f64) -> (i32){{$}}
; CHECK-NEXT: i32.trunc_s:sat/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
declare i32 @llvm.wasm.trunc.saturate.signed.i32.f64(double)
define i32 @i32_trunc_sat_s_f64(double %x) {
  %a = call i32 @llvm.wasm.trunc.saturate.signed.i32.f64(double %x)
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_u_f64:
; CHECK-NEXT: .functype i32_trunc_u_f64 (f64) -> (i32){{$}}
; CHECK-NEXT: i32.trunc_u:sat/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @i32_trunc_u_f64(double %x) {
  %a = fptoui double %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_sat_u_f64:
; CHECK-NEXT: .functype i32_trunc_sat_u_f64 (f64) -> (i32){{$}}
; CHECK-NEXT: i32.trunc_u:sat/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
declare i32 @llvm.wasm.trunc.saturate.unsigned.i32.f64(double)
define i32 @i32_trunc_sat_u_f64(double %x) {
  %a = call i32 @llvm.wasm.trunc.saturate.unsigned.i32.f64(double %x)
  ret i32 %a
}

; CHECK-LABEL: i64_trunc_s_f32:
; CHECK-NEXT: .functype i64_trunc_s_f32 (f32) -> (i64){{$}}
; CHECK-NEXT: i64.trunc_s:sat/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_trunc_s_f32(float %x) {
  %a = fptosi float %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_sat_s_f32:
; CHECK-NEXT: .functype i64_trunc_sat_s_f32 (f32) -> (i64){{$}}
; CHECK-NEXT: i64.trunc_s:sat/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
declare i64 @llvm.wasm.trunc.saturate.signed.i64.f32(float)
define i64 @i64_trunc_sat_s_f32(float %x) {
  %a = call i64 @llvm.wasm.trunc.saturate.signed.i64.f32(float %x)
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_u_f32:
; CHECK-NEXT: .functype i64_trunc_u_f32 (f32) -> (i64){{$}}
; CHECK-NEXT: i64.trunc_u:sat/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_trunc_u_f32(float %x) {
  %a = fptoui float %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_sat_u_f32:
; CHECK-NEXT: .functype i64_trunc_sat_u_f32 (f32) -> (i64){{$}}
; CHECK-NEXT: i64.trunc_u:sat/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
declare i64 @llvm.wasm.trunc.saturate.unsigned.i64.f32(float)
define i64 @i64_trunc_sat_u_f32(float %x) {
  %a = call i64 @llvm.wasm.trunc.saturate.unsigned.i64.f32(float %x)
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_s_f64:
; CHECK-NEXT: .functype i64_trunc_s_f64 (f64) -> (i64){{$}}
; CHECK-NEXT: i64.trunc_s:sat/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_trunc_s_f64(double %x) {
  %a = fptosi double %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_sat_s_f64:
; CHECK-NEXT: .functype i64_trunc_sat_s_f64 (f64) -> (i64){{$}}
; CHECK-NEXT: i64.trunc_s:sat/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
declare i64 @llvm.wasm.trunc.saturate.signed.i64.f64(double)
define i64 @i64_trunc_sat_s_f64(double %x) {
  %a = call i64 @llvm.wasm.trunc.saturate.signed.i64.f64(double %x)
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_u_f64:
; CHECK-NEXT: .functype i64_trunc_u_f64 (f64) -> (i64){{$}}
; CHECK-NEXT: i64.trunc_u:sat/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i64 @i64_trunc_u_f64(double %x) {
  %a = fptoui double %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_sat_u_f64:
; CHECK-NEXT: .functype i64_trunc_sat_u_f64 (f64) -> (i64){{$}}
; CHECK-NEXT: i64.trunc_u:sat/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
declare i64 @llvm.wasm.trunc.saturate.unsigned.i64.f64(double)
define i64 @i64_trunc_sat_u_f64(double %x) {
  %a = call i64 @llvm.wasm.trunc.saturate.unsigned.i64.f64(double %x)
  ret i64 %a
}

; CHECK-LABEL: f32_convert_s_i32:
; CHECK-NEXT: .functype f32_convert_s_i32 (i32) -> (f32){{$}}
; CHECK-NEXT: f32.convert_s/i32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @f32_convert_s_i32(i32 %x) {
  %a = sitofp i32 %x to float
  ret float %a
}

; CHECK-LABEL: f32_convert_u_i32:
; CHECK-NEXT: .functype f32_convert_u_i32 (i32) -> (f32){{$}}
; CHECK-NEXT: f32.convert_u/i32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @f32_convert_u_i32(i32 %x) {
  %a = uitofp i32 %x to float
  ret float %a
}

; CHECK-LABEL: f64_convert_s_i32:
; CHECK-NEXT: .functype f64_convert_s_i32 (i32) -> (f64){{$}}
; CHECK-NEXT: f64.convert_s/i32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @f64_convert_s_i32(i32 %x) {
  %a = sitofp i32 %x to double
  ret double %a
}

; CHECK-LABEL: f64_convert_u_i32:
; CHECK-NEXT: .functype f64_convert_u_i32 (i32) -> (f64){{$}}
; CHECK-NEXT: f64.convert_u/i32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @f64_convert_u_i32(i32 %x) {
  %a = uitofp i32 %x to double
  ret double %a
}

; CHECK-LABEL: f32_convert_s_i64:
; CHECK-NEXT: .functype f32_convert_s_i64 (i64) -> (f32){{$}}
; CHECK-NEXT: f32.convert_s/i64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @f32_convert_s_i64(i64 %x) {
  %a = sitofp i64 %x to float
  ret float %a
}

; CHECK-LABEL: f32_convert_u_i64:
; CHECK-NEXT: .functype f32_convert_u_i64 (i64) -> (f32){{$}}
; CHECK-NEXT: f32.convert_u/i64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @f32_convert_u_i64(i64 %x) {
  %a = uitofp i64 %x to float
  ret float %a
}

; CHECK-LABEL: f64_convert_s_i64:
; CHECK-NEXT: .functype f64_convert_s_i64 (i64) -> (f64){{$}}
; CHECK-NEXT: f64.convert_s/i64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @f64_convert_s_i64(i64 %x) {
  %a = sitofp i64 %x to double
  ret double %a
}

; CHECK-LABEL: f64_convert_u_i64:
; CHECK-NEXT: .functype f64_convert_u_i64 (i64) -> (f64){{$}}
; CHECK-NEXT: f64.convert_u/i64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @f64_convert_u_i64(i64 %x) {
  %a = uitofp i64 %x to double
  ret double %a
}

; CHECK-LABEL: f64_promote_f32:
; CHECK-NEXT: .functype f64_promote_f32 (f32) -> (f64){{$}}
; CHECK-NEXT: f64.promote/f32 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define double @f64_promote_f32(float %x) {
  %a = fpext float %x to double
  ret double %a
}

; CHECK-LABEL: f32_demote_f64:
; CHECK-NEXT: .functype f32_demote_f64 (f64) -> (f32){{$}}
; CHECK-NEXT: f32.demote/f64 $push[[NUM:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define float @f32_demote_f64(double %x) {
  %a = fptrunc double %x to float
  ret float %a
}

; If the high its are unused, LLVM will optimize sext/zext into anyext, which
; we need to patterm-match back to a specific instruction.

; CHECK-LABEL: anyext:
; CHECK: i64.extend_u/i32 $push0=, $0{{$}}
define i64 @anyext(i32 %x) {
    %y = sext i32 %x to i64
    %w = shl i64 %y, 32
    ret i64 %w
}

; CHECK-LABEL: bitcast_i32_to_float:
; CHECK: f32.reinterpret/i32   $push0=, $0{{$}}
define float @bitcast_i32_to_float(i32 %a) {
  %t = bitcast i32 %a to float
  ret float %t
}

; CHECK-LABEL: bitcast_float_to_i32:
; CHECK: i32.reinterpret/f32   $push0=, $0{{$}}
define i32 @bitcast_float_to_i32(float %a) {
  %t = bitcast float %a to i32
  ret i32 %t
}

; CHECK-LABEL: bitcast_i64_to_double:
; CHECK: f64.reinterpret/i64   $push0=, $0{{$}}
define double @bitcast_i64_to_double(i64 %a) {
  %t = bitcast i64 %a to double
  ret double %t
}

; CHECK-LABEL: bitcast_double_to_i64:
; CHECK: i64.reinterpret/f64   $push0=, $0{{$}}
define i64 @bitcast_double_to_i64(double %a) {
  %t = bitcast double %a to i64
  ret i64 %t
}
