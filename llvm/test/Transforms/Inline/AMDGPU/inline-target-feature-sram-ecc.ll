; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline < %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes='cgscc(inline)' < %s | FileCheck %s

; sram-ecc can be safely ignored when inlining, since no intrinisics
; or other directly exposed operations depend on it.

define i32 @func_default() #0 {
  ret i32 0
}

define i32 @func_ecc_enabled() #1 {
  ret i32 0
}

define i32 @func_ecc_disabled() #2 {
  ret i32 0
}

; CHECK-LABEL: @default_call_default(
; CHECK-NEXT: ret i32 0
define i32 @default_call_default() #0 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @ecc_enabled_call_default(
; CHECK-NEXT: ret i32 0
define i32 @ecc_enabled_call_default() #1 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @ecc_enabled_call_ecc_enabled(
; CHECK-NEXT: ret i32 0
define i32 @ecc_enabled_call_ecc_enabled() #1 {
  %call = call i32 @func_ecc_enabled()
  ret i32 %call
}

; CHECK-LABEL: @ecc_enabled_call_ecc_disabled(
; CHECK-NEXT: ret i32 0
define i32 @ecc_enabled_call_ecc_disabled() #1 {
  %call = call i32 @func_ecc_disabled()
  ret i32 %call
}

; CHECK-LABEL: @ecc_disabled_call_default(
; CHECK-NEXT: ret i32 0
define i32 @ecc_disabled_call_default() #2 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @ecc_disabled_call_ecc_enabled(
; CHECK-NEXT: ret i32 0
define i32 @ecc_disabled_call_ecc_enabled() #2 {
  %call = call i32 @func_ecc_enabled()
  ret i32 %call
}

; CHECK-LABEL: @ecc_disabled_call_ecc_disabled(
; CHECK-NEXT: ret i32 0
define i32 @ecc_disabled_call_ecc_disabled() #2 {
  %call = call i32 @func_ecc_disabled()
  ret i32 %call
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "target-features"="+sram-ecc" }
attributes #2 = { nounwind "target-features"="-sram-ecc" }
