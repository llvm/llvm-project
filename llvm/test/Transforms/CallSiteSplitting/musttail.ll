; RUN: opt < %s -passes=callsite-splitting -verify-dom-info -S | FileCheck %s

;CHECK-LABEL: @caller
;CHECK-LABEL: Top.split:
;CHECK: %ca1 = musttail call ptr @callee(ptr null, ptr %b)
;CHECK: %cb2 = bitcast ptr %ca1 to ptr
;CHECK: ret ptr %cb2
;CHECK-LABEL: TBB.split
;CHECK: %ca3 = musttail call ptr @callee(ptr nonnull %a, ptr null)
;CHECK: %cb4 = bitcast ptr %ca3 to ptr
;CHECK: ret ptr %cb4
define ptr @caller(ptr %a, ptr %b) {
Top:
  %c = icmp eq ptr %a, null
  br i1 %c, label %Tail, label %TBB
TBB:
  %c2 = icmp eq ptr %b, null
  br i1 %c2, label %Tail, label %End
Tail:
  %ca = musttail call ptr @callee(ptr %a, ptr %b)
  %cb = bitcast ptr %ca to ptr
  ret ptr %cb
End:
  ret ptr null
}

define ptr @callee(ptr %a, ptr %b) noinline {
  ret ptr %a
}

;CHECK-LABEL: @no_cast_caller
;CHECK-LABEL: Top.split:
;CHECK: %ca1 = musttail call ptr @callee(ptr null, ptr %b)
;CHECK: ret ptr %ca1
;CHECK-LABEL: TBB.split
;CHECK: %ca2 = musttail call ptr @callee(ptr nonnull %a, ptr null)
;CHECK: ret ptr %ca2
define ptr @no_cast_caller(ptr %a, ptr %b) {
Top:
  %c = icmp eq ptr %a, null
  br i1 %c, label %Tail, label %TBB
TBB:
  %c2 = icmp eq ptr %b, null
  br i1 %c2, label %Tail, label %End
Tail:
  %ca = musttail call ptr @callee(ptr %a, ptr %b)
  ret ptr %ca
End:
  ret ptr null
}

;CHECK-LABEL: @void_caller
;CHECK-LABEL: Top.split:
;CHECK: musttail call void @void_callee(ptr null, ptr %b)
;CHECK: ret void
;CHECK-LABEL: TBB.split
;CHECK: musttail call void @void_callee(ptr nonnull %a, ptr null)
;CHECK: ret void
define void @void_caller(ptr %a, ptr %b) {
Top:
  %c = icmp eq ptr %a, null
  br i1 %c, label %Tail, label %TBB
TBB:
  %c2 = icmp eq ptr %b, null
  br i1 %c2, label %Tail, label %End
Tail:
  musttail call void @void_callee(ptr %a, ptr %b)
  ret void
End:
  ret void
}

define void @void_callee(ptr %a, ptr %b) noinline {
  ret void
}

;   Include a test with a larger CFG that exercises the DomTreeUpdater
;   machinery a bit more.
;CHECK-LABEL: @larger_cfg_caller
;CHECK-LABEL: Top.split:
;CHECK: %r1 = musttail call ptr @callee(ptr null, ptr %b)
;CHECK: ret ptr %r1
;CHECK-LABEL: TBB.split
;CHECK: %r2 = musttail call ptr @callee(ptr nonnull %a, ptr null)
;CHECK: ret ptr %r2
define ptr @larger_cfg_caller(ptr %a, ptr %b) {
Top:
  %cond1 = icmp eq ptr %a, null
  br i1 %cond1, label %Tail, label %ExtraTest
ExtraTest:
  %a0 = load i8, ptr %a
  %cond2 = icmp eq i8 %a0, 0
  br i1 %cond2, label %TBB_pred, label %End
TBB_pred:
  br label %TBB
TBB:
  %cond3 = icmp eq ptr %b, null
  br i1 %cond3, label %Tail, label %End
Tail:
  %r = musttail call ptr @callee(ptr %a, ptr %b)
  ret ptr %r
End:
  ret ptr null
}
