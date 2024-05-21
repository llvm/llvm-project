; RUN: opt -passes="default<O1>" -S < %s | FileCheck %s
; RUN: opt -passes="thinlto-pre-link<O2>" -S < %s | FileCheck %s
; RUN: opt -passes="thinlto-pre-link<O2>,thinlto<O3>" -S < %s | FileCheck %s

target triple = "x86_64-unknown-linux"

define void @leaf_function() #0 {
entry:
  ret void
; CHECK-LABEL: entry:
; CHECK-NEXT:  %0 = tail call ptr @llvm.returnaddress(i32 0)
; CHECK-NEXT:  tail call void @__cyg_profile_func_enter(ptr nonnull @leaf_function, ptr %0)
; CHECK-NEXT:  tail call void @__cyg_profile_func_exit(ptr nonnull @leaf_function, ptr %0)
; CHECK-NEXT:  ret void
}


define void @root_function() #0 {
entry:
  call void @leaf_function()
  ret void
; CHECK-LABEL: entry:
; CHECK-NEXT:   %0 = tail call ptr @llvm.returnaddress(i32 0)
; CHECK-NEXT:   tail call void @__cyg_profile_func_enter(ptr nonnull @root_function, ptr %0)
; CHECK-NEXT:   tail call void @__cyg_profile_func_enter(ptr nonnull @leaf_function, ptr %0)
; CHECK-NEXT:   tail call void @__cyg_profile_func_exit(ptr nonnull @leaf_function, ptr %0)
; CHECK-NEXT:   tail call void @__cyg_profile_func_exit(ptr nonnull @root_function, ptr %0)
; CHECK-NEXT:   ret void
}

attributes #0 = { "instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit" }
