; RUN: opt -passes="function(ee-instrument),cgscc(inline),function(post-inline-ee-instrument)" -S < %s | FileCheck %s

; Running the passes twice should not result in more instrumentation.
; RUN: opt -passes="function(ee-instrument),function(ee-instrument),cgscc(inline),function(post-inline-ee-instrument),function(post-inline-ee-instrument)" -S < %s | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux"

define void @leaf_function() #0 {
entry:
  ret void

; CHECK-LABEL: define void @leaf_function()
; CHECK: entry:
; CHECK-NEXT: call void @mcount()
; CHECK-NEXT: %0 = call i8* @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_enter(i8* bitcast (void ()* @leaf_function to i8*), i8* %0)
; CHECK-NEXT: %1 = call i8* @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_exit(i8* bitcast (void ()* @leaf_function to i8*), i8* %1)
; CHECK-NEXT: ret void
}


define void @root_function() #0 {
entry:
  call void @leaf_function()
  ret void

; CHECK-LABEL: define void @root_function()
; CHECK: entry:
; CHECK-NEXT: call void @mcount()

; CHECK-NEXT: %0 = call i8* @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_enter(i8* bitcast (void ()* @root_function to i8*), i8* %0)

; Entry and exit calls, inlined from @leaf_function()
; CHECK-NEXT: %1 = call i8* @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_enter(i8* bitcast (void ()* @leaf_function to i8*), i8* %1)
; CHECK-NEXT: %2 = call i8* @llvm.returnaddress(i32 0)
; CHECK-NEXT: call void @__cyg_profile_func_exit(i8* bitcast (void ()* @leaf_function to i8*), i8* %2)
; CHECK-NEXT: %3 = call i8* @llvm.returnaddress(i32 0)

; CHECK-NEXT: call void @__cyg_profile_func_exit(i8* bitcast (void ()* @root_function to i8*), i8* %3)
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
declare i32* @tailcallee()
define i32* @tailcaller() #8 {
  %1 = musttail call i32* @tailcallee()
  ret i32* %1
; CHECK-LABEL: define i32* @tailcaller
; CHECK: call void @__cyg_profile_func_exit
; CHECK: musttail call i32* @tailcallee
; CHECK: ret
}
define i8* @tailcaller2() #8 {
  %1 = musttail call i32* @tailcallee()
  %2 = bitcast i32* %1 to i8*
  ret i8* %2
; CHECK-LABEL: define i8* @tailcaller2
; CHECK: call void @__cyg_profile_func_exit
; CHECK: musttail call i32* @tailcallee
; CHECK: bitcast
; CHECK: ret
}

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
