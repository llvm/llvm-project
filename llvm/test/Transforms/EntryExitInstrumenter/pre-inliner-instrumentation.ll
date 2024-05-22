; RUN: opt -passes="default<O0>" -S < %s | FileCheck -check-prefix=PRELTO %s
; RUN: opt -passes="default<O1>" -S < %s | FileCheck -check-prefix=PRELTO %s
; RUN: opt -passes="thinlto-pre-link<O0>,thinlto<O0>" -S < %s | FileCheck -check-prefix=PRELTO %s
; RUN: opt -passes="thinlto-pre-link<O2>" -S < %s | FileCheck -check-prefix=PRELTO %s
; RUN: opt -passes="thinlto<O2>" -S < %s | FileCheck -check-prefix=LTO %s
; RUN: opt -passes="lto<O2>" -S < %s | FileCheck -check-prefix=LTO %s

; Pre-inline instrumentation should be inserted, but not by LTO/ThinLTO passes.

target triple = "x86_64-unknown-linux"

define void @leaf_function() #0 {
entry:
  ret void
; PRELTO-LABEL: entry:
; PRELTO-NEXT:  %0 ={{( tail)?}} call ptr @llvm.returnaddress(i32 0)
; PRELTO-NEXT:  {{( tail)? call void @__cyg_profile_func_enter\(ptr( nonnull)? @leaf_function, ptr %0\)}}
; LTO-NOT:      {{( tail)?}} call void @__cyg_profile_func_enter
; PRELTO:       {{( tail)?}} call void @__cyg_profile_func_exit
; PRELTO-NEXT:  ret void
; LTO-NOT:      {{( tail)?}} call void @__cyg_profile_func_exit
; LTO-LABEL:    entry:
; LTO-NEXT:     ret void
}


define void @root_function() #1 {
entry:
  call void @leaf_function()
  ret void
; PRELTO-LABEL: entry:
; PRELTO-NEXT:  %0 ={{( tail)?}} call ptr @llvm.returnaddress(i32 0)
; PRELTO-NEXT:  {{( tail)?}} call void @__cyg_profile_func_enter(ptr{{( nonnull)?}} @root_function, ptr %0)
; PRELTO:       {{( tail)?}} call void @__cyg_profile_func_enter
; PRELTO:       {{( tail)?}} call void @__cyg_profile_func_exit
; PRELTO:       {{( tail)?}} call void @__cyg_profile_func_exit
; PRELTO-NEXT:  ret void
; LTO-LABEL:    entry:
; LTO-NEXT:     ret void
; LTO-NOT:      {{( tail)?}} call void @__cyg_profile_func_exit
}

attributes #0 = { alwaysinline "instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit" }
attributes #1 = { "instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit" }
