; RUN: opt -passes="function(ee-instrument),cgscc(inline),function(ee-instrument<post-inline>)" -S < %s | FileCheck %s

; Running the passes twice should not result in more instrumentation.
; RUN: opt -passes="function(ee-instrument),function(ee-instrument),cgscc(inline),function(ee-instrument<post-inline>),function(ee-instrument<post-inline>)" -S < %s | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux"

define void @leaf_function() #0 {
entry:
  ret void

; CHECK-LABEL: define void @leaf_function()
; CHECK: entry:
; CHECK-NEXT: call void @mcount()
; CHECK-NEXT: %0 = call ptr @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_enter(ptr @leaf_function, ptr %0)
; CHECK-NEXT: %1 = call ptr @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_exit(ptr @leaf_function, ptr %1)
; CHECK-NEXT: ret void
}


define void @root_function() #0 {
entry:
  call void @leaf_function()
  ret void

; CHECK-LABEL: define void @root_function()
; CHECK: entry:
; CHECK-NEXT: call void @mcount()

; CHECK-NEXT: %0 = call ptr @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_enter(ptr @root_function, ptr %0)

; Entry and exit calls, inlined from @leaf_function()
; CHECK-NEXT: %1 = call ptr @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_enter(ptr @leaf_function, ptr %1)
; CHECK-NEXT: %2 = call ptr @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_exit(ptr @leaf_function, ptr %2)
; CHECK-NEXT: %3 = call ptr @llvm.returnaddress(i32 0)

; CHECK-NEXT: call void @__cyg_profile_func_exit(ptr @root_function, ptr %3)
; CHECK-NEXT: ret void
}



; The mcount function has many different names.

define void @f1() #1 { entry: ret void }
; CHECK-LABEL: define void @f1
; CHECK: call void @.mcount

define void @f2() #2 { entry: ret void }
; CHECK-LABEL: define void @f2
; CHECK: call void @llvm.arm.gnu.eabi.mcount

define void @f3() #3 { entry: ret void }
; CHECK-LABEL: define void @f3
; CHECK: call void @"\01_mcount"

define void @f4() #4 { entry: ret void }
; CHECK-LABEL: define void @f4
; CHECK: call void @"\01mcount"

define void @f5() #5 { entry: ret void }
; CHECK-LABEL: define void @f5
; CHECK: call void @__mcount

define void @f6() #6 { entry: ret void }
; CHECK-LABEL: define void @f6
; CHECK: call void @_mcount

define void @f7() #7 { entry: ret void }
; CHECK-LABEL: define void @f7
; CHECK: call void @__cyg_profile_func_enter_bare


; Treat musttail calls as terminators; inserting between the musttail call and
; ret is not allowed.
declare ptr @tailcallee()
define ptr @tailcaller() #8 {
  %1 = musttail call ptr @tailcallee()
  ret ptr %1
; CHECK-LABEL: define ptr @tailcaller
; CHECK: call void @__cyg_profile_func_exit
; CHECK: musttail call ptr @tailcallee
; CHECK: ret
}
define ptr @tailcaller2() #8 {
  %1 = musttail call ptr @tailcallee()
  %2 = bitcast ptr %1 to ptr
  ret ptr %2
; CHECK-LABEL: define ptr @tailcaller2
; CHECK: call void @__cyg_profile_func_exit
; CHECK: musttail call ptr @tailcallee
; CHECK: bitcast
; CHECK: ret
}

;; naked functions are not instrumented, otherwise the argument registers
;; and the return address register (if present) would be clobbered.
define void @naked() naked { entry: ret void }
; CHECK-LABEL:      define void @naked(
; CHECK-LABEL-NEXT: entry:
; CHECK-LABEL-NEXT:   ret void

; The attributes are "consumed" when the instrumentation is inserted.
; CHECK: attributes
; CHECK-NOT: instrument-function

attributes #0 = { "instrument-function-entry-inlined"="mcount" "instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit" }
attributes #1 = { "instrument-function-entry-inlined"=".mcount" }
attributes #2 = { "instrument-function-entry-inlined"="llvm.arm.gnu.eabi.mcount" }
attributes #3 = { "instrument-function-entry-inlined"="\01_mcount" }
attributes #4 = { "instrument-function-entry-inlined"="\01mcount" }
attributes #5 = { "instrument-function-entry-inlined"="__mcount" }
attributes #6 = { "instrument-function-entry-inlined"="_mcount" }
attributes #7 = { "instrument-function-entry-inlined"="__cyg_profile_func_enter_bare" }
attributes #8 = { "instrument-function-exit"="__cyg_profile_func_exit" }
