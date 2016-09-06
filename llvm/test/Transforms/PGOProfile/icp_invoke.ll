; RUN: opt < %s -icp-lto -pgo-icall-prom -S -icp-count-threshold=0 | FileCheck %s --check-prefix=ICP
; RUN: opt < %s -icp-lto -passes=pgo-icall-prom -S -icp-count-threshold=0 | FileCheck %s --check-prefix=ICP
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo1 = global void ()* null, align 8
@foo2 = global i32 ()* null, align 8
@_ZTIi = external constant i8*

define internal void @_ZL4bar1v() !PGOFuncName !0 {
entry:
  ret void
}

define internal i32 @_ZL4bar2v() !PGOFuncName !1 {
entry:
  ret i32 100
}

define i32 @_Z3goov() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %tmp = load void ()*, void ()** @foo1, align 8
; ICP:  [[BITCAST_IC1:%[0-9]+]] = bitcast void ()* %tmp to i8*
; ICP:  [[CMP_IC1:%[0-9]+]] = icmp eq i8* [[BITCAST_IC1]], bitcast (void ()* @_ZL4bar1v to i8*)
; ICP:  br i1 [[CMP_IC1]], label %[[TRUE_LABEL_IC1:.*]], label %[[FALSE_LABEL_IC1:.*]], !prof [[BRANCH_WEIGHT:![0-9]+]]
; ICP:[[TRUE_LABEL_IC1]]:
; ICP:  invoke void @_ZL4bar1v()
; ICP:          to label %[[DCALL_NORMAL_DEST_IC1:.*]] unwind label %lpad
; ICP:[[FALSE_LABEL_IC1]]:
  invoke void %tmp()
          to label %try.cont unwind label %lpad, !prof !2

; ICP:[[DCALL_NORMAL_DEST_IC1]]:
; ICP:  br label %try.cont

lpad:
  %tmp1 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %tmp2 = extractvalue { i8*, i32 } %tmp1, 0
  %tmp3 = extractvalue { i8*, i32 } %tmp1, 1
  %tmp4 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %tmp3, %tmp4
  br i1 %matches, label %catch, label %eh.resume

catch:
  %tmp5 = tail call i8* @__cxa_begin_catch(i8* %tmp2)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  %tmp6 = load i32 ()*, i32 ()** @foo2, align 8
; ICP:  [[BITCAST_IC2:%[0-9]+]] = bitcast i32 ()* %tmp6 to i8*
; ICP:  [[CMP_IC2:%[0-9]+]] = icmp eq i8* [[BITCAST_IC2]], bitcast (i32 ()* @_ZL4bar2v to i8*)
; ICP:  br i1 [[CMP_IC2]], label %[[TRUE_LABEL_IC2:.*]], label %[[FALSE_LABEL_IC2:.*]], !prof [[BRANCH_WEIGHT:![0-9]+]]
; ICP:[[TRUE_LABEL_IC2]]:
; ICP:  [[RESULT_IC2:%[0-9]+]] = invoke i32 @_ZL4bar2v()
; ICP:          to label %[[DCALL_NORMAL_DEST_IC2:.*]] unwind label %lpad1
; ICP:[[FALSE_LABEL_IC2]]:
  %call = invoke i32 %tmp6()
          to label %try.cont8 unwind label %lpad1, !prof !3

; ICP:[[DCALL_NORMAL_DEST_IC2]]:
; ICP:  br label %try.cont8
lpad1:
  %tmp7 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %tmp8 = extractvalue { i8*, i32 } %tmp7, 0
  %tmp9 = extractvalue { i8*, i32 } %tmp7, 1
  %tmp10 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches5 = icmp eq i32 %tmp9, %tmp10
  br i1 %matches5, label %catch6, label %eh.resume

catch6:
  %tmp11 = tail call i8* @__cxa_begin_catch(i8* %tmp8)
  tail call void @__cxa_end_catch()
  br label %try.cont8

try.cont8:
  %i.0 = phi i32 [ undef, %catch6 ], [ %call, %try.cont ]
; ICP:  %i.0 = phi i32 [ undef, %catch6 ], [ %call, %[[FALSE_LABEL_IC2]] ], [ [[RESULT_IC2]], %[[DCALL_NORMAL_DEST_IC2]] ]
  ret i32 %i.0

eh.resume:
  %ehselector.slot.0 = phi i32 [ %tmp9, %lpad1 ], [ %tmp3, %lpad ]
  %exn.slot.0 = phi i8* [ %tmp8, %lpad1 ], [ %tmp2, %lpad ]
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.0, 0
  %lpad.val11 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { i8*, i32 } %lpad.val11
}

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

!0 = !{!"invoke.ll:_ZL4bar1v"}
!1 = !{!"invoke.ll:_ZL4bar2v"}
!2 = !{!"VP", i32 0, i64 1, i64 -2732222848796217051, i64 1}
!3 = !{!"VP", i32 0, i64 1, i64 -6116256810522035449, i64 1}
; ICP-NOT !3 = !{!"VP", i32 0, i64 1, i64 -2732222848796217051, i64 1}
; ICP-NOT !4 = !{!"VP", i32 0, i64 1, i64 -6116256810522035449, i64 1}
; ICP: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 1, i32 0}
