; RUN: opt < %s -icp-lto -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICP
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTISt9exception = external constant ptr
@pfptr = global ptr null, align 8

define internal i32 @_ZL4bar1v() !PGOFuncName !0 {
entry:
  ret i32 100 
}

; Function Attrs: uwtable
define i32 @_Z3fooi(i32 %x) local_unnamed_addr personality ptr @__gxx_personality_v0 {
entry:
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %if.end, label %cleanup

if.end:                                           ; preds = %entry
  %fptr = load ptr, ptr @pfptr, align 8
; ICP:  [[CMP_IC1:%[0-9]+]] = icmp eq ptr %fptr, @_ZL4bar1v
; ICP:  br i1 [[CMP_IC1]], label %[[TRUE_LABEL_IC1:.*]], label %[[FALSE_LABEL_IC1:.*]], !prof [[BRANCH_WEIGHT:![0-9]+]]
; ICP:[[TRUE_LABEL_IC1]]:
; ICP:  invoke i32 @_ZL4bar1v()
; ICP:          to label %[[DCALL_NORMAL_DEST_IC1:.*]] unwind label %lpad
; ICP:[[FALSE_LABEL_IC1]]:
  %call = invoke i32 %fptr()
          to label %cleanup unwind label %lpad, !prof !1

; ICP:[[DCALL_NORMAL_DEST_IC1]]:
; ICP:  br label %cleanup

lpad:                                             ; preds = %if.end
  %0 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTISt9exception
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTISt9exception)
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %ehcleanup

catch:                                            ; preds = %lpad
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3)
  tail call void @__cxa_end_catch()
  br label %cleanup

cleanup:                                          ; preds = %catch, %if.end, %entry
; ICP-NOT: %[0-9]+ = phi 
  ret i32 0

ehcleanup:                                        ; preds = %lpad
  resume { ptr, i32 } %0
}

declare i32 @_Z3barv() local_unnamed_addr

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr)

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

!0 = !{!"invoke.ll:_ZL4bar1v"}
!1 = !{!"VP", i32 0, i64 10000, i64 -2732222848796217051, i64 10000}
