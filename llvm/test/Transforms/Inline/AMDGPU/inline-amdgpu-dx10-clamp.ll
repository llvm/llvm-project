; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -inline < %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes='cgscc(inline)' < %s | FileCheck %s

define i32 @func_default() #0 {
  ret i32 0
}

define i32 @func_dx10_clamp_enabled() #1 {
  ret i32 0
}

define i32 @func_dx10_clamp_disabled() #2 {
  ret i32 0
}

; CHECK-LABEL: @default_call_default(
; CHECK-NEXT: ret i32 0
define i32 @default_call_default() #0 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @dx10_clamp_enabled_call_default(
; CHECK-NEXT: ret i32 0
define i32 @dx10_clamp_enabled_call_default() #1 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @dx10_clamp_enabled_call_dx10_clamp_enabled(
; CHECK-NEXT: ret i32 0
define i32 @dx10_clamp_enabled_call_dx10_clamp_enabled() #1 {
  %call = call i32 @func_dx10_clamp_enabled()
  ret i32 %call
}

; CHECK-LABEL: @dx10_clamp_enabled_call_dx10_clamp_disabled(
; CHECK-NEXT: call i32 @func_dx10_clamp_disabled()
define i32 @dx10_clamp_enabled_call_dx10_clamp_disabled() #1 {
  %call = call i32 @func_dx10_clamp_disabled()
  ret i32 %call
}

; CHECK-LABEL: @dx10_clamp_disabled_call_default(
; CHECK-NEXT: call i32 @func_default()
define i32 @dx10_clamp_disabled_call_default() #2 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @dx10_clamp_disabled_call_dx10_clamp_enabled(
; CHECK-NEXT: call i32 @func_dx10_clamp_enabled()
define i32 @dx10_clamp_disabled_call_dx10_clamp_enabled() #2 {
  %call = call i32 @func_dx10_clamp_enabled()
  ret i32 %call
}

; CHECK-LABEL: @dx10_clamp_disabled_call_dx10_clamp_disabled(
; CHECK-NEXT: ret i32 0
define i32 @dx10_clamp_disabled_call_dx10_clamp_disabled() #2 {
  %call = call i32 @func_dx10_clamp_disabled()
  ret i32 %call
}

; Shader calling a compute function
; CHECK-LABEL: @amdgpu_ps_default_call_default(
; CHECK-NEXT: call i32 @func_default()
define amdgpu_ps i32 @amdgpu_ps_default_call_default() #0 {
  %call = call i32 @func_default()
  ret i32 %call
}

; Shader with dx10_clamp enabled calling a compute function. Default
; also implies ieee_mode, so this isn't inlinable.
; CHECK-LABEL: @amdgpu_ps_dx10_clamp_enabled_call_default(
; CHECK-NEXT: call i32 @func_default()
define amdgpu_ps i32 @amdgpu_ps_dx10_clamp_enabled_call_default() #1 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @amdgpu_ps_dx10_clamp_disabled_call_default(
; CHECK-NEXT: call i32 @func_default()
define amdgpu_ps i32 @amdgpu_ps_dx10_clamp_disabled_call_default() #2 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @amdgpu_ps_dx10_clamp_enabled_ieee_call_default(
; CHECK-NEXT: ret i32 0
define amdgpu_ps i32 @amdgpu_ps_dx10_clamp_enabled_ieee_call_default() #3 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @amdgpu_ps_dx10_clamp_disabled_ieee_call_default(
; CHECK-NEXT: call i32 @func_default()
define amdgpu_ps i32 @amdgpu_ps_dx10_clamp_disabled_ieee_call_default() #4 {
  %call = call i32 @func_default()
  ret i32 %call
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-dx10-clamp"="true" }
attributes #2 = { nounwind "amdgpu-dx10-clamp"="false" }
attributes #3 = { nounwind "amdgpu-dx10-clamp"="true" "amdgpu-ieee"="true" }
attributes #4 = { nounwind "amdgpu-dx10-clamp"="false" "amdgpu-ieee"="true" }
