; RUN: opt -S -passes=deadargelim %s | FileCheck %s

; Case 0: the basic example: an entire aggregate use is returned, but it's
; actually only used in ways we can eliminate. We gain benefit from analysing
; the "use" and applying its results to all sub-values.

; CHECK-LABEL: define internal void @agguse_dead()

define internal { i32, i32 } @agguse_dead() {
  ret { i32, i32 } { i32 0, i32 1 }
}

define internal { i32, i32 } @test_agguse_dead() {
  %val = call { i32, i32 } @agguse_dead()
  ret { i32, i32 } %val
}



; Case 1: an opaque use of the aggregate exists (in this case dead). Otherwise
; only one value is used, so function can be simplified.

; CHECK-LABEL: define internal i32 @rets_independent_if_agguse_dead()
; CHECK: [[RET:%.*]] = extractvalue { i32, i32 } { i32 0, i32 1 }, 1
; CHECK: ret i32 [[RET]]

define internal { i32, i32 } @rets_independent_if_agguse_dead() {
  ret { i32, i32 } { i32 0, i32 1 }
}

define internal { i32, i32 } @test_rets_independent_if_agguse_dead(i1 %tst) {
  %val = call { i32, i32 } @rets_independent_if_agguse_dead()
  br i1 %tst, label %use_1, label %use_aggregate

use_1:
  ; This use can be classified as applying only to ret 1.
  %val0 = extractvalue { i32, i32 } %val, 1
  call void @callee(i32 %val0)
  ret { i32, i32 } poison

use_aggregate:
  ; This use is assumed to apply to both 0 and 1.
  ret { i32, i32 } %val
}

; Case 2: an opaque use of the aggregate exists (in this case *live*). Other
; uses shouldn't matter.

; CHECK-LABEL: define internal { i32, i32 } @rets_live_agguse()
; CHECK: ret { i32, i32 } { i32 0, i32 1 }

define internal { i32, i32 } @rets_live_agguse() {
  ret { i32, i32} { i32 0, i32 1 }
}

define { i32, i32 } @test_rets_live_aggues(i1 %tst) {
  %val = call { i32, i32 } @rets_live_agguse()
  br i1 %tst, label %use_1, label %use_aggregate

use_1:
  ; This use can be classified as applying only to ret 1.
  %val0 = extractvalue { i32, i32 } %val, 1
  call void @callee(i32 %val0)
  ret { i32, i32 } poison

use_aggregate:
  ; This use is assumed to apply to both 0 and 1.
  ret { i32, i32 } %val
}

declare void @callee(i32)

; Case 3: the insertvalue meant %in was live if ret-slot-1 was, but we were only
; tracking multiple ret-slots for struct types. So %in was eliminated
; incorrectly.

; CHECK-LABEL: define internal [2 x i32] @array_rets_have_multiple_slots(i32 %in)

define internal [2 x i32] @array_rets_have_multiple_slots(i32 %in) {
  %ret = insertvalue [2 x i32] poison, i32 %in, 1
  ret [2 x i32] %ret
}

define [2 x i32] @test_array_rets_have_multiple_slots() {
  %res = call [2 x i32] @array_rets_have_multiple_slots(i32 42)
  ret [2 x i32] %res
}

; Case 4: we can remove some retvals from the array. It's nice to produce an
; array again having done so (rather than converting it to a struct).

; CHECK-LABEL: define internal [2 x i32] @can_shrink_arrays()
; CHECK: [[VAL0:%.*]] = extractvalue [3 x i32] [i32 42, i32 43, i32 44], 0
; CHECK: [[RESTMP:%.*]] = insertvalue [2 x i32] poison, i32 [[VAL0]], 0
; CHECK: [[VAL2:%.*]] = extractvalue [3 x i32] [i32 42, i32 43, i32 44], 2
; CHECK: [[RES:%.*]] = insertvalue [2 x i32] [[RESTMP]], i32 [[VAL2]], 1
; CHECK: ret [2 x i32] [[RES]]

; CHECK-LABEL: define void @test_can_shrink_arrays()

define internal [3 x i32] @can_shrink_arrays() {
  ret [3 x i32] [i32 42, i32 43, i32 44]
}

define void @test_can_shrink_arrays() {
  %res = call [3 x i32] @can_shrink_arrays()

  %res.0 = extractvalue [3 x i32] %res, 0
  call void @callee(i32 %res.0)

  %res.2 = extractvalue [3 x i32] %res, 2
  call void @callee(i32 %res.2)

  ret void
}

; Case 5: %in gets passed directly to the return. It should mark be marked as
; used if *any* of the return values are, not just if value 0 is.

; CHECK-LABEL: define internal i32 @ret_applies_to_all({ i32, i32 } %in)
; CHECK: [[RET:%.*]] = extractvalue { i32, i32 } %in, 1
; CHECK: ret i32 [[RET]]

define internal {i32, i32} @ret_applies_to_all({i32, i32} %in) {
  ret {i32, i32} %in
}

define i32 @test_ret_applies_to_all() {
  %val = call {i32, i32} @ret_applies_to_all({i32, i32} {i32 42, i32 43})
  %ret = extractvalue {i32, i32} %val, 1
  ret i32 %ret
}

; Case 6: When considering @mid, the return instruciton has sub-value 0
; unconditionally live, but 1 only conditionally live. Since at that level we're
; applying the results to the whole of %res, this means %res is live and cannot
; be reduced. There is scope for further optimisation here (though not visible
; in this test-case).

; CHECK-LABEL: define internal { ptr, i32 } @inner()

define internal {ptr, i32} @mid() {
  %res = call {ptr, i32} @inner()
  %intval = extractvalue {ptr, i32} %res, 1
  %tst = icmp eq i32 %intval, 42
  br i1 %tst, label %true, label %true

true:
  ret {ptr, i32} %res
}

define internal {ptr, i32} @inner() {
  ret {ptr, i32} {ptr null, i32 42}
}

define internal i8 @outer() {
  %res = call {ptr, i32} @mid()
  %resptr = extractvalue {ptr, i32} %res, 0

  %val = load i8, ptr %resptr
  ret i8 %val
}

define internal { i32 } @agg_ret() {
entry:
  unreachable
}

; CHECK-LABEL: define void @PR24906
; CHECK: %[[invoke:.*]] = invoke i32 @agg_ret()
; CHECK: %[[oldret:.*]] = insertvalue { i32 } poison, i32 %[[invoke]], 0
; CHECK: phi { i32 } [ %[[oldret]],
define void @PR24906() personality ptr poison {
entry:
  %tmp2 = invoke { i32 } @agg_ret()
          to label %bb3 unwind label %bb4

bb3:
  %tmp3 = phi { i32 } [ %tmp2, %entry ]
  unreachable

bb4:
  %tmp4 = landingpad { ptr, i32 }
          cleanup
  unreachable
}
