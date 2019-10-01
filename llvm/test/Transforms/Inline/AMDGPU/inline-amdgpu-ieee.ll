; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -inline < %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes='cgscc(inline)' < %s | FileCheck %s

define i32 @func_default() #0 {
  ret i32 0
}

define i32 @func_ieee_enabled() #1 {
  ret i32 0
}

define i32 @func_ieee_disabled() #2 {
  ret i32 0
}

; CHECK-LABEL: @default_call_default(
; CHECK-NEXT: ret i32 0
define i32 @default_call_default() #0 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @ieee_enabled_call_default(
; CHECK-NEXT: ret i32 0
define i32 @ieee_enabled_call_default() #1 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @ieee_enabled_call_ieee_enabled(
; CHECK-NEXT: ret i32 0
define i32 @ieee_enabled_call_ieee_enabled() #1 {
  %call = call i32 @func_ieee_enabled()
  ret i32 %call
}

; CHECK-LABEL: @ieee_enabled_call_ieee_disabled(
; CHECK-NEXT: call i32 @func_ieee_disabled()
define i32 @ieee_enabled_call_ieee_disabled() #1 {
  %call = call i32 @func_ieee_disabled()
  ret i32 %call
}

; CHECK-LABEL: @ieee_disabled_call_default(
; CHECK-NEXT: call i32 @func_default()
define i32 @ieee_disabled_call_default() #2 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @ieee_disabled_call_ieee_enabled(
; CHECK-NEXT: call i32 @func_ieee_enabled()
define i32 @ieee_disabled_call_ieee_enabled() #2 {
  %call = call i32 @func_ieee_enabled()
  ret i32 %call
}

; CHECK-LABEL: @ieee_disabled_call_ieee_disabled(
; CHECK-NEXT: ret i32 0
define i32 @ieee_disabled_call_ieee_disabled() #2 {
  %call = call i32 @func_ieee_disabled()
  ret i32 %call
}

; Shader calling a compute function
; CHECK-LABEL: @amdgpu_ps_default_call_default(
; CHECK-NEXT: call i32 @func_default()
define amdgpu_ps i32 @amdgpu_ps_default_call_default() #0 {
  %call = call i32 @func_default()
  ret i32 %call
}

; Shader with ieee enabled calling a compute function
; CHECK-LABEL: @amdgpu_ps_ieee_enabled_call_default(
; CHECK-NEXT: ret i32 0
define amdgpu_ps i32 @amdgpu_ps_ieee_enabled_call_default() #1 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @amdgpu_ps_ieee_disabled_call_default(
; CHECK-NEXT: call i32 @func_default()
define amdgpu_ps i32 @amdgpu_ps_ieee_disabled_call_default() #2 {
  %call = call i32 @func_default()
  ret i32 %call
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-ieee"="true" }
attributes #2 = { nounwind "amdgpu-ieee"="false" }
