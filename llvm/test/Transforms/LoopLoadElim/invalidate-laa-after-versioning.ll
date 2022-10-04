; RUN: opt -passes='loop-vectorize,loop-load-elim' -S %s | FileCheck %s

; REQUIRES: asserts
; XFAIL: *

@glob.1 = external global [100 x double]
@glob.2 = external global [100 x double]

; Test for PR57825 to make sure LAA is properly invalidated after versioning
; loops.
define void @test(ptr %arg, i64 %arg1) {
; CHECK-LABEL: @test
;
bb:
  br label %outer.header

outer.header:                                              ; preds = %bb21, %bb
  %ptr.phi = phi ptr [ %arg, %bb ], [ @glob.1, %outer.latch ]
  %gep.1 = getelementptr inbounds double, ptr %ptr.phi, i64 3
  br label %inner.1

inner.1:
  %iv.1 = phi i64 [ 0, %outer.header ], [ %iv.next, %inner.1 ]
  %ptr.iv.1 = phi ptr [ @glob.2, %outer.header ], [ %ptr.iv.1.next, %inner.1 ]
  %tmp25 = mul nuw nsw i64 %iv.1, %arg1
  %gep.2 = getelementptr inbounds double, ptr %gep.1, i64 %tmp25
  store double 0.000000e+00, ptr %gep.2, align 8
  %gep.3 = getelementptr double, ptr %ptr.phi, i64 %tmp25
  %gep.4 = getelementptr double, ptr %gep.3, i64 2
  %tmp29 = load double, ptr %gep.4, align 8
  %ptr.iv.1.next = getelementptr inbounds double, ptr %ptr.iv.1, i64 1
  %iv.next = add nuw nsw i64 %iv.1, 1
  %c.1 = icmp eq i64 %iv.1, 1
  br i1 %c.1, label %inner.1.exit, label %inner.1

inner.1.exit:                                              ; preds = %bb22
  %lcssa.ptr.iv.1 = phi ptr [ %ptr.iv.1, %inner.1 ]
  %gep.5 = getelementptr inbounds double, ptr %lcssa.ptr.iv.1, i64 1
  br label %inner.2

inner.2:
  %ptr.iv.2 = phi ptr [ %gep.5, %inner.1.exit ], [ %ptr.iv.2.next, %inner.2 ]
  %ptr.iv.2.next = getelementptr inbounds double, ptr %ptr.iv.2, i64 1
  br i1 false, label %inner.2.exit, label %inner.2

inner.2.exit:
  %lcssa.ptr.iv.2 = phi ptr [ %ptr.iv.2, %inner.2 ]
  %gep.6 = getelementptr inbounds double, ptr %ptr.phi, i64 1
  %gep.7 = getelementptr inbounds double, ptr %lcssa.ptr.iv.2, i64 1
  br label %inner.3

inner.3:                                             ; preds = %bb14, %bb10
  %iv.2 = phi i64 [ 0, %inner.2.exit ], [ %iv.2.next, %inner.3 ]
  %gep.8 = getelementptr inbounds double, ptr %gep.6, i64 %iv.2
  store double 0.000000e+00, ptr %gep.7, align 8
  store double 0.000000e+00, ptr %gep.8, align 8
  %gep.9 = getelementptr double, ptr %ptr.phi, i64 %iv.2
  %tmp18 = load double, ptr %gep.9, align 8
  %iv.2.next = add nuw nsw i64 %iv.2, 1
  %c.2 = icmp eq i64 %iv.2, 1
  br i1 %c.2, label %outer.latch, label %inner.3

outer.latch:
  br label %outer.header
}
