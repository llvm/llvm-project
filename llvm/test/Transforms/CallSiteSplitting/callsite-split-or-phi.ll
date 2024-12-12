; RUN: opt < %s -passes=callsite-splitting -S | FileCheck %s
; RUN: opt < %s -passes='function(callsite-splitting)' -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

;CHECK-LABEL: @test_eq_eq

;CHECK-LABEL: Header:
;CHECK: br i1 %tobool1, label %Header.split, label %TBB
;CHECK-LABEL: Header.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr null, i32 %v, i32 1)
;CHECK-LABEL: TBB:
;CHECK: br i1 %cmp, label %TBB.split, label %End
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr nonnull %a, i32 1, i32 2)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_eq_eq(ptr %a, i32 %v) {
Header:
  %tobool1 = icmp eq ptr %a, null
  br i1 %tobool1, label %Tail, label %TBB

TBB:
  %cmp = icmp eq i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_eq_eq_eq
;CHECK-LABEL: Header2.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr %a, i32 %v, i32 10)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 1, i32 %p)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header2.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_eq_eq_eq(ptr %a, i32 %v, i32 %p) {
Header:
  %tobool1 = icmp eq ptr %a, null
  br i1 %tobool1, label %Header2, label %End

Header2:
  %tobool2 = icmp eq i32 %p, 10
  br i1 %tobool2, label %Tail, label %TBB

TBB:
  %cmp = icmp eq i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_eq_eq_eq_constrain_same_i32_arg
;CHECK-LABEL: Header2.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr %a, i32 222, i32 %p)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 333, i32 %p)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header2.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_eq_eq_eq_constrain_same_i32_arg(ptr %a, i32 %v, i32 %p) {
Header:
  %tobool1 = icmp eq i32 %v, 111
  br i1 %tobool1, label %Header2, label %End

Header2:
  %tobool2 = icmp eq i32 %v, 222
  br i1 %tobool2, label %Tail, label %TBB

TBB:
  %cmp = icmp eq i32 %v, 333
  br i1 %cmp, label %Tail, label %End

Tail:
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_ne_eq
;CHECK-LABEL: Header.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr nonnull %a, i32 %v, i32 1)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr null, i32 1, i32 2)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_ne_eq(ptr %a, i32 %v) {
Header:
  %tobool1 = icmp ne ptr %a, null
  br i1 %tobool1, label %Tail, label %TBB

TBB:
  %cmp = icmp eq i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_ne_eq_ne
;CHECK-LABEL: Header2.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr nonnull %a, i32 %v, i32 10)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 %v, i32 %p)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header2.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_ne_eq_ne(ptr %a, i32 %v, i32 %p) {
Header:
  %tobool1 = icmp ne ptr %a, null
  br i1 %tobool1, label %Header2, label %TBB

Header2:
  %tobool2 = icmp eq i32 %p, 10
  br i1 %tobool2, label %Tail, label %TBB

TBB:
  %cmp = icmp ne i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_ne_ne
;CHECK-LABEL: Header.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr nonnull %a, i32 %v, i32 1)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr null, i32 %v, i32 2)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_ne_ne(ptr %a, i32 %v) {
Header:
  %tobool1 = icmp ne ptr %a, null
  br i1 %tobool1, label %Tail, label %TBB

TBB:
  %cmp = icmp ne i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_ne_ne_ne_constrain_same_pointer_arg
;CHECK-LABEL: Header2.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr nonnull %a, i32 %v, i32 %p)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 %v, i32 %p)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header2.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_ne_ne_ne_constrain_same_pointer_arg(ptr %a, i32 %v, i32 %p, ptr %a2, ptr %a3) {
Header:
  %tobool1 = icmp ne ptr %a, null
  br i1 %tobool1, label %Header2, label %TBB

Header2:
  %tobool2 = icmp ne ptr %a, %a2
  br i1 %tobool2, label %Tail, label %TBB

TBB:
  %cmp = icmp ne ptr %a, %a3
  br i1 %cmp, label %Tail, label %End

Tail:
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}



;CHECK-LABEL: @test_eq_eq_untaken
;CHECK-LABEL: Header.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr nonnull %a, i32 %v, i32 1)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr null, i32 1, i32 2)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_eq_eq_untaken(ptr %a, i32 %v) {
Header:
  %tobool1 = icmp eq ptr %a, null
  br i1 %tobool1, label %TBB, label %Tail

TBB:
  %cmp = icmp eq i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_eq_eq_eq_untaken
;CHECK-LABEL: Header2.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr nonnull %a, i32 %v, i32 10)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 1, i32 %p)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header2.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_eq_eq_eq_untaken(ptr %a, i32 %v, i32 %p) {
Header:
  %tobool1 = icmp eq ptr %a, null
  br i1 %tobool1, label %TBB, label %Header2

Header2:
  %tobool2 = icmp eq i32 %p, 10
  br i1 %tobool2, label %Tail, label %TBB

TBB:
  %cmp = icmp eq i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_ne_eq_untaken
;CHECK-LABEL: Header.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr null, i32 %v, i32 1)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr nonnull %a, i32 1, i32 2)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_ne_eq_untaken(ptr %a, i32 %v) {
Header:
  %tobool1 = icmp ne ptr %a, null
  br i1 %tobool1, label %TBB, label %Tail

TBB:
  %cmp = icmp eq i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_ne_eq_ne_untaken
;CHECK-LABEL: Header2.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr null, i32 %v, i32 10)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 %v, i32 %p)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header2.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_ne_eq_ne_untaken(ptr %a, i32 %v, i32 %p) {
Header:
  %tobool1 = icmp ne ptr %a, null
  br i1 %tobool1, label %TBB, label %Header2

Header2:
  %tobool2 = icmp eq i32 %p, 10
  br i1 %tobool2, label %Tail, label %TBB

TBB:
  %cmp = icmp ne i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_ne_ne_untaken
;CHECK-LABEL: Header.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr null, i32 %v, i32 1)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr nonnull %a, i32 1, i32 2)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_ne_ne_untaken(ptr %a, i32 %v) {
Header:
  %tobool1 = icmp ne ptr %a, null
  br i1 %tobool1, label %TBB, label %Tail

TBB:
  %cmp = icmp ne i32 %v, 1
  br i1 %cmp, label %End, label %Tail

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_nonconst_const_phi
;CHECK-LABEL: Header.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr %a, i32 %v, i32 1)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 1, i32 2)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_nonconst_const_phi(ptr %a, ptr %b, i32 %v) {
Header:
  %tobool1 = icmp eq ptr %a, %b
  br i1 %tobool1, label %Tail, label %TBB

TBB:
  %cmp = icmp eq i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_nonconst_nonconst_phi
;CHECK-LABEL: Header.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr %a, i32 %v, i32 1)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 %v, i32 2)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL2]], %TBB.split ], [ %[[CALL1]], %Header.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_nonconst_nonconst_phi(ptr %a, ptr %b, i32 %v, i32 %v2) {
Header:
  %tobool1 = icmp eq ptr %a, %b
  br i1 %tobool1, label %Tail, label %TBB

TBB:
  %cmp = icmp eq i32 %v, %v2
  br i1 %cmp, label %Tail, label %End

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_cfg_no_or_phi
;CHECK-LABEL: TBB0.split
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr %a, i32 %v, i32 1)
;CHECK-LABEL: TBB1.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 %v, i32 2)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL2]], %TBB1.split ], [ %[[CALL1]], %TBB0.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_cfg_no_or_phi(ptr %a,  i32 %v, i1 %arg) {
entry:
  br i1 %arg, label %TBB0, label %TBB1
TBB0:
  br i1 %arg, label %Tail, label %End
TBB1:
  br i1 %arg, label %Tail, label %End
Tail:
  %p = phi i32[1,%TBB0], [2, %TBB1]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r
End:
  ret i32 %v
}

;CHECK-LABEL: @test_nonconst_nonconst_phi_noncost
;CHECK-NOT: Header.split:
;CHECK-NOT: TBB.split:
;CHECK-LABEL: Tail:
;CHECK: %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
;CHECK: ret i32 %r
define i32 @test_nonconst_nonconst_phi_noncost(ptr %a, ptr %b, i32 %v, i32 %v2) {
Header:
  %tobool1 = icmp eq ptr %a, %b
  br i1 %tobool1, label %Tail, label %TBB

TBB:
  %cmp = icmp eq i32 %v, %v2
  br i1 %cmp, label %Tail, label %End

Tail:
  %p = phi i32[%v,%Header], [%v2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_3preds_constphi
;CHECK-NOT: Header.split:
;CHECK-NOT: TBB.split:
;CHECK-LABEL: Tail:
;CHECK: %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
;CHECK: ret i32 %r
define i32 @test_3preds_constphi(ptr %a, i32 %v, i1 %c1, i1 %c2, i1 %c3) {
Header:
  br i1 %c1, label %Tail, label %TBB1

TBB1:
  br i1 %c2, label %Tail, label %TBB2

TBB2:
  br i1 %c3, label %Tail, label %End

Tail:
  %p = phi i32[1,%Header], [2, %TBB1], [3, %TBB2]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_indirectbr_phi
;CHECK-NOT: Header.split:
;CHECK-NOT: TBB.split:
;CHECK-LABEL: Tail:
;CHECK: %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
;CHECK: ret i32 %r
define i32 @test_indirectbr_phi(ptr %address, ptr %a, ptr %b, i32 %v) {
Header:
   %indirect.goto.dest = select i1 undef, ptr blockaddress(@test_indirectbr_phi, %End), ptr %address
   indirectbr ptr %indirect.goto.dest, [label %TBB, label %Tail]

TBB:
  %indirect.goto.dest2 = select i1 undef, ptr blockaddress(@test_indirectbr_phi, %End), ptr %address
  indirectbr ptr %indirect.goto.dest2, [label %Tail, label %End]

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_cond_no_effect
;CHECK-NOT: Header.split:
;CHECK-NOT: TBB.split:
;CHECK-LABEL: Tail:
;CHECK: %r = call i32 @callee(ptr %a, i32 %v, i32 0)
;CHECK: ret i32 %r
define i32 @test_cond_no_effect(ptr %a, i32 %v, i1 %arg) {
Entry:
  %tobool1 = icmp eq ptr %a, null
  br i1 %tobool1, label %Header, label %End

Header:
  br i1 %arg, label %Tail, label %TBB

TBB:
  br i1 %arg, label %Tail, label %End

Tail:
  %r = call i32 @callee(ptr %a, i32 %v, i32 0)
  ret i32 %r

End:
  ret i32 %v
}

;CHECK-LABEL: @test_unreachable
;CHECK-LABEL: Header.split:
;CHECK: %[[CALL1:.*]] = call i32 @callee(ptr %a, i32 %v, i32 10)
;CHECK-LABEL: TBB.split:
;CHECK: %[[CALL2:.*]] = call i32 @callee(ptr %a, i32 1, i32 %p)
;CHECK-LABEL: Tail
;CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header.split ], [ %[[CALL2]], %TBB.split ]
;CHECK: ret i32 %[[MERGED]]
define i32 @test_unreachable(ptr %a, i32 %v, i32 %p) {
Entry:
  br label %End
Header:
  %tobool2 = icmp eq i32 %p, 10
  br i1 %tobool2, label %Tail, label %TBB
TBB:
  %cmp = icmp eq i32 %v, 1
  br i1 %cmp, label %Tail, label %Header
Tail:
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r
End:
  ret i32 %v
}

define i32 @callee(ptr %a, i32 %v, i32 %p) {
entry:
  %c = icmp ne ptr %a, null
  br i1 %c, label %BB1, label %BB2

BB1:
  call void @dummy(ptr %a, i32 %p)
  br label %End

BB2:
  call void @dummy2(i32 %v, i32 %p)
  br label %End

End:
  ret i32 %p
}

declare void @dummy(ptr, i32)
declare void @dummy2(i32, i32)

; Make sure we remove the non-nullness on constant paramater.
;
;CHECK-LABEL: @caller2
;CHECK-LABEL: Top1.split:
;CHECK: call i32 @callee(ptr inttoptr (i64 4643 to ptr)
define void @caller2(i32 %c, ptr %a_elt, ptr %b_elt) {
entry:
  br label %Top0

Top0:
  %tobool1 = icmp eq ptr %a_elt, inttoptr (i64 4643 to  ptr)
  br i1 %tobool1, label %Top1, label %NextCond

Top1:
  %tobool2 = icmp ne ptr %a_elt, null
  br i1 %tobool2, label %CallSiteBB, label %NextCond

NextCond:
  %cmp = icmp ne ptr %b_elt, null
  br i1 %cmp, label %CallSiteBB, label %End

CallSiteBB:
  call i32 @callee(ptr %a_elt, i32 %c, i32 %c)
  br label %End

End:
  ret void
}

; CHECK-LABEL: i32 @test_multiple_phis(
; CHECK:      Header.split:
; CHECK-NEXT:   %r2 = call i32 @callee(ptr null, i32 1, i32 5)
; CHECK-NEXT:   br label %Tail

; CHECK:      TBB.split:
; CHECK-NEXT:   %r1 = call i32 @callee(ptr null, i32 2, i32 10)
; CHECK-NEXT:   br label %Tail

; CHECK:        Tail:
; CHECK-NEXT:   %phi.call = phi i32 [ %r1, %TBB.split ], [ %r2, %Header.split ]
; CHECK-NEXT:   %p.0 = phi i32 [ 0, %Header.split ], [ 99, %TBB.split ]
; CHECK-NEXT:   %res = add i32 %phi.call, %p.0
; CHECK-NEXT:   ret i32 %phi.call
;
define i32 @test_multiple_phis(i1 %c.1) {
Header:
  br i1 %c.1, label %Tail, label %TBB

TBB:
  br label %Tail

Tail:
  %p.0 = phi i32 [0, %Header], [99, %TBB]
  %p.1 = phi i32[1, %Header], [2, %TBB]
  %p.2 = phi i32 [5, %Header], [10, %TBB]
  %r = call i32 @callee(ptr null, i32 %p.1, i32 %p.2)
  %res = add i32 %r, %p.0
  ret i32 %r

End:
  ret i32 10
}
