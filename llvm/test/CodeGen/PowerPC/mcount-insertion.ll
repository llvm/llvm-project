; RUN: opt -ee-instrument < %s | opt -inline | opt -post-inline-ee-instrument | llc -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s

; The run-line mimics how Clang might run the instrumentation passes.

target datalayout = "E-m:e-i64:64-n32:64"


define void @leaf_function() #0 {
entry:
  ret void

; CHECK-LABEL: leaf_function:
; CHECK: bl mcount
; CHECK-NOT: bl
; CHECK: bl __cyg_profile_func_enter
; CHECK-NOT: bl
; CHECK: bl __cyg_profile_func_exit
; CHECK-NOT: bl
; CHECK: blr
}


define void @root_function() #0 {
entry:
  call void @leaf_function()
  ret void

; CHECK-LABEL: root_function:
; CHECK: bl mcount
; CHECK-NOT: bl
; CHECK: bl __cyg_profile_func_enter
; CHECK-NOT: bl

; Entry and exit calls, inlined from @leaf_function()
; CHECK: bl __cyg_profile_func_enter
; CHECK-NOT: bl
; CHECK: bl __cyg_profile_func_exit
; CHECK-NOT: bl

; CHECK: bl __cyg_profile_func_exit
; CHECK-NOT: bl
; CHECK: blr
}

attributes #0 = { "instrument-function-entry-inlined"="mcount" "instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit" }
