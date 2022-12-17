; RUN: opt < %s -passes=callsite-splitting -verify-dom-info -S | FileCheck %s

;CHECK-LABEL: @caller
;CHECK-LABEL: Top.split:
;CHECK: %ca1 = musttail call i8* @callee(i8* null, i8* %b)
;CHECK: %cb2 = bitcast i8* %ca1 to i8*
;CHECK: ret i8* %cb2
;CHECK-LABEL: TBB.split
;CHECK: %ca3 = musttail call i8* @callee(i8* nonnull %a, i8* null)
;CHECK: %cb4 = bitcast i8* %ca3 to i8*
;CHECK: ret i8* %cb4
define i8* @caller(i8* %a, i8* %b) {
Top:
  %c = icmp eq i8* %a, null
  br i1 %c, label %Tail, label %TBB
TBB:
  %c2 = icmp eq i8* %b, null
  br i1 %c2, label %Tail, label %End
Tail:
  %ca = musttail call i8* @callee(i8* %a, i8* %b)
  %cb = bitcast i8* %ca to i8*
  ret i8* %cb
End:
  ret i8* null
}

define i8* @callee(i8* %a, i8* %b) noinline {
  ret i8* %a
}

;CHECK-LABEL: @no_cast_caller
;CHECK-LABEL: Top.split:
;CHECK: %ca1 = musttail call i8* @callee(i8* null, i8* %b)
;CHECK: ret i8* %ca1
;CHECK-LABEL: TBB.split
;CHECK: %ca2 = musttail call i8* @callee(i8* nonnull %a, i8* null)
;CHECK: ret i8* %ca2
define i8* @no_cast_caller(i8* %a, i8* %b) {
Top:
  %c = icmp eq i8* %a, null
  br i1 %c, label %Tail, label %TBB
TBB:
  %c2 = icmp eq i8* %b, null
  br i1 %c2, label %Tail, label %End
Tail:
  %ca = musttail call i8* @callee(i8* %a, i8* %b)
  ret i8* %ca
End:
  ret i8* null
}

;CHECK-LABEL: @void_caller
;CHECK-LABEL: Top.split:
;CHECK: musttail call void @void_callee(i8* null, i8* %b)
;CHECK: ret void
;CHECK-LABEL: TBB.split
;CHECK: musttail call void @void_callee(i8* nonnull %a, i8* null)
;CHECK: ret void
define void @void_caller(i8* %a, i8* %b) {
Top:
  %c = icmp eq i8* %a, null
  br i1 %c, label %Tail, label %TBB
TBB:
  %c2 = icmp eq i8* %b, null
  br i1 %c2, label %Tail, label %End
Tail:
  musttail call void @void_callee(i8* %a, i8* %b)
  ret void
End:
  ret void
}

define void @void_callee(i8* %a, i8* %b) noinline {
  ret void
}

;   Include a test with a larger CFG that exercises the DomTreeUpdater
;   machinery a bit more.
;CHECK-LABEL: @larger_cfg_caller
;CHECK-LABEL: Top.split:
;CHECK: %r1 = musttail call i8* @callee(i8* null, i8* %b)
;CHECK: ret i8* %r1
;CHECK-LABEL: TBB.split
;CHECK: %r2 = musttail call i8* @callee(i8* nonnull %a, i8* null)
;CHECK: ret i8* %r2
define i8* @larger_cfg_caller(i8* %a, i8* %b) {
Top:
  %cond1 = icmp eq i8* %a, null
  br i1 %cond1, label %Tail, label %ExtraTest
ExtraTest:
  %a0 = load i8, i8* %a
  %cond2 = icmp eq i8 %a0, 0
  br i1 %cond2, label %TBB_pred, label %End
TBB_pred:
  br label %TBB
TBB:
  %cond3 = icmp eq i8* %b, null
  br i1 %cond3, label %Tail, label %End
Tail:
  %r = musttail call i8* @callee(i8* %a, i8* %b)
  ret i8* %r
End:
  ret i8* null
}
