; RUN: opt < %s -callsite-splitting -S | FileCheck %s
; RUN: opt < %s  -passes='function(callsite-splitting)' -S | FileCheck %s

; CHECK-LABEL: @test_simple
; CHECK-LABEL: Header:
; CHECK-NEXT: br i1 undef, label %Tail.predBB.split
; CHECK-LABEL: TBB:
; CHECK: br i1 %cmp, label %Tail.predBB.split1
; CHECK-LABEL: Tail.predBB.split:
; CHECK: %[[CALL1:.*]] = call i32 @callee(i32* %a, i32 %v, i32 %p)
; CHECK-LABEL: Tail.predBB.split1:
; CHECK: %[[CALL2:.*]] = call i32 @callee(i32* null, i32 %v, i32 %p)
; CHECK-LABEL: Tail
; CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Tail.predBB.split ], [ %[[CALL2]], %Tail.predBB.split1 ]
; CHECK: ret i32 %[[MERGED]]
define i32 @test_simple(i32* %a, i32 %v, i32 %p) {
Header:
  br i1 undef, label %Tail, label %End

TBB:
  %cmp = icmp eq i32* %a, null
  br i1 %cmp, label %Tail, label %End

Tail:
  %r = call i32 @callee(i32* %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

; CHECK-LABEL: @test_eq_eq_eq_untaken
; CHECK-LABEL: Header:
; CHECK: br i1 %tobool1, label %TBB1, label %Tail.predBB.split
; CHECK-LABEL: TBB2:
; CHECK: br i1 %cmp2, label %Tail.predBB.split1, label %End
; CHECK-LABEL: Tail.predBB.split:
; CHECK: %[[CALL1:.*]] = call i32 @callee(i32* nonnull %a, i32 %v, i32 %p)
; CHECK-LABEL: Tail.predBB.split1:
; CHECK: %[[CALL2:.*]] = call i32 @callee(i32* null, i32 1, i32 99)
; CHECK-LABEL: Tail
; CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Tail.predBB.split ], [ %[[CALL2]], %Tail.predBB.split1 ]
; CHECK: ret i32 %[[MERGED]]
define i32 @test_eq_eq_eq_untaken2(i32* %a, i32 %v, i32 %p) {
Header:
  %tobool1 = icmp eq i32* %a, null
  br i1 %tobool1, label %TBB1, label %Tail

TBB1:
  %cmp1 = icmp eq i32 %v, 1
  br i1 %cmp1, label %TBB2, label %End

TBB2:
  %cmp2 = icmp eq i32 %p, 99
  br i1 %cmp2, label %Tail, label %End

Tail:
  %r = call i32 @callee(i32* %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

; CHECK-LABEL: @test_eq_ne_eq_untaken
; CHECK-LABEL: Header:
; CHECK: br i1 %tobool1, label %TBB1, label %Tail.predBB.split
; CHECK-LABEL: TBB2:
; CHECK: br i1 %cmp2, label %Tail.predBB.split1, label %End
; CHECK-LABEL: Tail.predBB.split:
; CHECK: %[[CALL1:.*]] = call i32 @callee(i32* nonnull %a, i32 %v, i32 %p)
; CHECK-LABEL: Tail.predBB.split1:
; CHECK: %[[CALL2:.*]] = call i32 @callee(i32* null, i32 %v, i32 99)
; CHECK-LABEL: Tail
; CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Tail.predBB.split ], [ %[[CALL2]], %Tail.predBB.split1 ]
; CHECK: ret i32 %[[MERGED]]
define i32 @test_eq_ne_eq_untaken(i32* %a, i32 %v, i32 %p) {
Header:
  %tobool1 = icmp eq i32* %a, null
  br i1 %tobool1, label %TBB1, label %Tail

TBB1:
  %cmp1 = icmp ne i32 %v, 1
  br i1 %cmp1, label %TBB2, label %End

TBB2:
  %cmp2 = icmp eq i32 %p, 99
  br i1 %cmp2, label %Tail, label %End

Tail:
  %r = call i32 @callee(i32* %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

; CHECK-LABEL: @test_header_header2_tbb
; CHECK: Header2:
; CHECK:br i1 %tobool2, label %Tail.predBB.split, label %TBB1
; CHECK-LABEL: TBB2:
; CHECK: br i1 %cmp2, label %Tail.predBB.split1, label %End
; CHECK-LABEL: Tail.predBB.split:
; CHECK: %[[CALL1:.*]] = call i32 @callee(i32* nonnull %a, i32 %v, i32 10)
; CHECK-LABEL: Tail.predBB.split1:
; NOTE: CallSiteSplitting cannot infer that %a is null here, as it currently
;       only supports recording conditions along a single predecessor path.
; CHECK: %[[CALL2:.*]] = call i32 @callee(i32* %a, i32 1, i32 99)
; CHECK-LABEL: Tail
; CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Tail.predBB.split ], [ %[[CALL2]], %Tail.predBB.split1 ]
; CHECK: ret i32 %[[MERGED]]
define i32 @test_header_header2_tbb(i32* %a, i32 %v, i32 %p) {
Header:
  %tobool1 = icmp eq i32* %a, null
  br i1 %tobool1, label %TBB1, label %Header2

Header2:
  %tobool2 = icmp eq i32 %p, 10
  br i1 %tobool2, label %Tail, label %TBB1

TBB1:
  %cmp1 = icmp eq i32 %v, 1
  br i1 %cmp1, label %TBB2, label %End

TBB2:
  %cmp2 = icmp eq i32 %p, 99
  br i1 %cmp2, label %Tail, label %End

Tail:
  %r = call i32 @callee(i32* %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}

define i32 @callee(i32* %a, i32 %v, i32 %p) {
  ret i32 10
}
