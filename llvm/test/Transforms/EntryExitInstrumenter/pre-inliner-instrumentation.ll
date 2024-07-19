; RUN: opt -passes="default<O0>" -S < %s | FileCheck -check-prefix=INSTRUMENT %s
; RUN: opt -passes="default<O1>" -S < %s | FileCheck -check-prefix=INSTRUMENT %s
; RUN: opt -passes="thinlto-pre-link<O0>" -S < %s | FileCheck -check-prefix=INSTRUMENT %s
; RUN: opt -passes="thinlto-pre-link<O2>" -S < %s | FileCheck -check-prefix=INSTRUMENT %s
; RUN: opt -passes="thinlto<O0>" -S < %s | FileCheck -check-prefix=NOINSTRUMENT %s
; RUN: opt -passes="thinlto<O2>" -S < %s | FileCheck -check-prefix=NOINSTRUMENT %s
; RUN: opt -passes="lto<O0>" -S < %s | FileCheck -check-prefix=NOINSTRUMENT %s
; RUN: opt -passes="lto<O2>" -S < %s | FileCheck -check-prefix=NOINSTRUMENT %s

; Pre-inline instrumentation should be inserted, but not by LTO/ThinLTO passes.

target triple = "x86_64-unknown-linux"

define void @leaf_function() #0 {
entry:
  ret void
; INSTRUMENT-LABEL:   entry:
; INSTRUMENT-NEXT:    %0 ={{.*}} call ptr @llvm.returnaddress(i32 0)
; INSTRUMENT-NEXT:    {{.* call void @__cyg_profile_func_enter\(ptr( nonnull)? @leaf_function, ptr %0\)}}
; NOINSTRUMENT-NOT:   {{.*}} call void @__cyg_profile_func_enter
; INSTRUMENT:         {{.*}} call void @__cyg_profile_func_exit
; INSTRUMENT-NEXT:    ret void
; NOINSTRUMENT-NOT:   {{.*}} call void @__cyg_profile_func_exit
; NOINSTRUMENT-LABEL: entry:
; NOINSTRUMENT-NEXT:  ret void
}


define void @root_function() #1 {
entry:
  call void @leaf_function()
  ret void
; INSTRUMENT-LABEL:   entry:
; INSTRUMENT-NEXT:    %0 ={{.*}} call ptr @llvm.returnaddress(i32 0)
; INSTRUMENT-NEXT:    {{.*}} call void @__cyg_profile_func_enter(ptr{{( nonnull)?}} @root_function, ptr %0)
; INSTRUMENT:         {{.*}} call void @__cyg_profile_func_enter
; INSTRUMENT:         {{.*}} call void @__cyg_profile_func_exit
; INSTRUMENT:         {{.*}} call void @__cyg_profile_func_exit
; INSTRUMENT-NEXT:    ret void
; NOINSTRUMENT-LABEL: entry:
; NOINSTRUMENT:       ret void
; NOINSTRUMENT-NOT:   {{.*}} call void @__cyg_profile_func_exit
}

attributes #0 = { alwaysinline "instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit" }
attributes #1 = { "instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit" }
