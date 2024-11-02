; RUN: opt < %s -icp-lto -passes=pgo-icall-prom -S | FileCheck %s --check-prefix=ICP
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo1 = global ptr null, align 8
@foo2 = global ptr null, align 8
@_ZTIi = external constant ptr

define internal void @_ZL4bar1v() !PGOFuncName !0 {
entry:
  ret void
}

define internal i32 @_ZL4bar2v() !PGOFuncName !1 {
entry:
  ret i32 100
}

define i32 @_Z3goov() personality ptr @__gxx_personality_v0 {
entry:
  %tmp = load ptr, ptr @foo1, align 8
; ICP:  [[CMP_IC1:%[0-9]+]] = icmp eq ptr %tmp, @_ZL4bar1v
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
  %tmp1 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %tmp2 = extractvalue { ptr, i32 } %tmp1, 0
  %tmp3 = extractvalue { ptr, i32 } %tmp1, 1
  %tmp4 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %tmp3, %tmp4
  br i1 %matches, label %catch, label %eh.resume

catch:
  %tmp5 = tail call ptr @__cxa_begin_catch(ptr %tmp2)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  %tmp6 = load ptr, ptr @foo2, align 8
; ICP:  [[CMP_IC2:%[0-9]+]] = icmp eq ptr %tmp6, @_ZL4bar2v
; ICP:  br i1 [[CMP_IC2]], label %[[TRUE_LABEL_IC2:.*]], label %[[FALSE_LABEL_IC2:.*]], !prof [[BRANCH_WEIGHT:![0-9]+]]
; ICP:[[TRUE_LABEL_IC2]]:
; ICP:  [[RESULT_IC2_0:%[0-9]+]] = invoke i32 @_ZL4bar2v()
; ICP:          to label %[[MERGE_BB:.*]] unwind label %lpad1
; ICP:[[FALSE_LABEL_IC2]]:
; ICP:  [[RESULT_IC2_1:%.+]] = invoke i32 %tmp6()
; ICP:          to label %[[MERGE_BB]] unwind label %lpad1
  %call = invoke i32 %tmp6()
          to label %try.cont8 unwind label %lpad1, !prof !3

; ICP:[[MERGE_BB]]:
; ICP:  [[MERGE_PHI:%.+]] = phi i32 [ [[RESULT_IC2_1]], %[[FALSE_LABEL_IC2]] ], [ [[RESULT_IC2_0]], %[[TRUE_LABEL_IC2]] ]
; ICP:  br label %try.cont8
lpad1:
  %tmp7 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %tmp8 = extractvalue { ptr, i32 } %tmp7, 0
  %tmp9 = extractvalue { ptr, i32 } %tmp7, 1
  %tmp10 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches5 = icmp eq i32 %tmp9, %tmp10
  br i1 %matches5, label %catch6, label %eh.resume

catch6:
  %tmp11 = tail call ptr @__cxa_begin_catch(ptr %tmp8)
  tail call void @__cxa_end_catch()
  br label %try.cont8

try.cont8:
  %i.0 = phi i32 [ undef, %catch6 ], [ %call, %try.cont ]
; ICP:  %i.0 = phi i32 [ undef, %catch6 ], [ [[MERGE_PHI]], %[[MERGE_BB]] ]
  ret i32 %i.0

eh.resume:
  %ehselector.slot.0 = phi i32 [ %tmp9, %lpad1 ], [ %tmp3, %lpad ]
  %exn.slot.0 = phi ptr [ %tmp8, %lpad1 ], [ %tmp2, %lpad ]
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn.slot.0, 0
  %lpad.val11 = insertvalue { ptr, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { ptr, i32 } %lpad.val11
}

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(ptr)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

!0 = !{!"invoke.ll:_ZL4bar1v"}
!1 = !{!"invoke.ll:_ZL4bar2v"}
!2 = !{!"VP", i32 0, i64 1, i64 -2732222848796217051, i64 1}
!3 = !{!"VP", i32 0, i64 1, i64 -6116256810522035449, i64 1}
; ICP-NOT: !3 = !{!"VP", i32 0, i64 1, i64 -2732222848796217051, i64 1}
; ICP-NOT: !4 = !{!"VP", i32 0, i64 1, i64 -6116256810522035449, i64 1}
; ICP: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 1, i32 0}
