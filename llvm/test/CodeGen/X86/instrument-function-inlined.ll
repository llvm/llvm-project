; RUN: llc -mtriple=x86_64-- -O0 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-- -O1 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-- -O2 < %s | FileCheck %s

; The codegen should insert post-inlining instrumentation calls and should not
; insert pre-inlining instrumentation.

; CHECK-NOT:       callq __cyg_profile_func_enter

define void @leaf_function() #0 {
; CHECK-LABEL: leaf_function:
; CHECK:       callq __cyg_profile_func_enter_bare
; CHECK:       callq __cyg_profile_func_exit
  ret void
}

define void @root_function() #0 {
entry:
; CHECK-LABEL: root_function:
; CHECK:       callq __cyg_profile_func_enter_bare
; CHECK-NEXT:  callq leaf_function
; CHECK:       callq __cyg_profile_func_exit
  call void @leaf_function()
  ret void
}

attributes #0 = { "instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-entry-inlined"="__cyg_profile_func_enter_bare" "instrument-function-exit-inlined"="__cyg_profile_func_exit" }
