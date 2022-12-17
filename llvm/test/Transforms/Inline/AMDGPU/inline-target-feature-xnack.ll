; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline < %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes='cgscc(inline)' < %s | FileCheck %s

define i32 @func_default() #0 {
  ret i32 0
}

define i32 @func_xnack_enabled() #1 {
  ret i32 0
}

define i32 @func_xnack_disabled() #2 {
  ret i32 0
}

; CHECK-LABEL: @default_call_default(
; CHECK-NEXT: ret i32 0
define i32 @default_call_default() #0 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @xnack_enabled_call_default(
; CHECK-NEXT: ret i32 0
define i32 @xnack_enabled_call_default() #1 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @xnack_enabled_call_xnack_enabled(
; CHECK-NEXT: ret i32 0
define i32 @xnack_enabled_call_xnack_enabled() #1 {
  %call = call i32 @func_xnack_enabled()
  ret i32 %call
}

; CHECK-LABEL: @xnack_enabled_call_xnack_disabled(
; CHECK-NEXT: ret i32 0
define i32 @xnack_enabled_call_xnack_disabled() #1 {
  %call = call i32 @func_xnack_disabled()
  ret i32 %call
}

; CHECK-LABEL: @xnack_disabled_call_default(
; CHECK-NEXT: ret i32 0
define i32 @xnack_disabled_call_default() #2 {
  %call = call i32 @func_default()
  ret i32 %call
}

; CHECK-LABEL: @xnack_disabled_call_xnack_enabled(
; CHECK-NEXT: ret i32 0
define i32 @xnack_disabled_call_xnack_enabled() #2 {
  %call = call i32 @func_xnack_enabled()
  ret i32 %call
}

; CHECK-LABEL: @xnack_disabled_call_xnack_disabled(
; CHECK-NEXT: ret i32 0
define i32 @xnack_disabled_call_xnack_disabled() #2 {
  %call = call i32 @func_xnack_disabled()
  ret i32 %call
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "target-features"="+xnack" }
attributes #2 = { nounwind "target-features"="-xnack" }
