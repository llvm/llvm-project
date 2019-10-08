; RUN: opt -attributor -attributor-manifest-internal --attributor-disable=false -attributor-max-iterations-verify -attributor-max-iterations=1 -S < %s | FileCheck %s --check-prefixes=ATTRIBUTOR


declare void @deref_phi_user(i32* %a);

; TEST 1
; take mininimum of return values
;
define i32* @test1(i32* dereferenceable(4) %0, double* dereferenceable(8) %1, i1 zeroext %2) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(4) i32* @test1(i32* nonnull readnone dereferenceable(4) "no-capture-maybe-returned" %0, double* nonnull readnone dereferenceable(8) "no-capture-maybe-returned" %1, i1 zeroext %2)
  %4 = bitcast double* %1 to i32*
  %5 = select i1 %2, i32* %0, i32* %4
  ret i32* %5
}

; TEST 2
define i32* @test2(i32* dereferenceable_or_null(4) %0, double* dereferenceable(8) %1, i1 zeroext %2) local_unnamed_addr {
; ATTRIBUTOR: define dereferenceable_or_null(4) i32* @test2(i32* readnone dereferenceable_or_null(4) "no-capture-maybe-returned" %0, double* nonnull readnone dereferenceable(8) "no-capture-maybe-returned" %1, i1 zeroext %2)
  %4 = bitcast double* %1 to i32*
  %5 = select i1 %2, i32* %0, i32* %4
  ret i32* %5
}

; TEST 3
; GEP inbounds
define i32* @test3_1(i32* dereferenceable(8) %0) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(4) i32* @test3_1(i32* nonnull readnone dereferenceable(8) "no-capture-maybe-returned" %0)
  %ret = getelementptr inbounds i32, i32* %0, i64 1
  ret i32* %ret
}

define i32* @test3_2(i32* dereferenceable_or_null(32) %0) local_unnamed_addr {
; FIXME: Argument should be mark dereferenceable because of GEP `inbounds`.
; ATTRIBUTOR: define nonnull dereferenceable(16) i32* @test3_2(i32* readnone dereferenceable_or_null(32) "no-capture-maybe-returned" %0)
  %ret = getelementptr inbounds i32, i32* %0, i64 4
  ret i32* %ret
}

define i32* @test3_3(i32* dereferenceable(8) %0, i32* dereferenceable(16) %1, i1 %2) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(4) i32* @test3_3(i32* nonnull readnone dereferenceable(8) "no-capture-maybe-returned" %0, i32* nonnull readnone dereferenceable(16) "no-capture-maybe-returned" %1, i1 %2) local_unnamed_addr
  %ret1 = getelementptr inbounds i32, i32* %0, i64 1
  %ret2 = getelementptr inbounds i32, i32* %1, i64 2
  %ret = select i1 %2, i32* %ret1, i32* %ret2
  ret i32* %ret
}

; TEST 4
; Better than known in IR.

define dereferenceable(4) i32* @test4(i32* dereferenceable(8) %0) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(8) i32* @test4(i32* nonnull readnone returned dereferenceable(8) "no-capture-maybe-returned" %0)
  ret i32* %0
}

; TEST 5
; loop in which dereferenceabily "grows"
define void @deref_phi_growing(i32* dereferenceable(4000) %a) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %a.addr.0 = phi i32* [ %a, %entry ], [ %incdec.ptr, %for.inc ]
; CHECK: call void @deref_phi_user(i32* dereferenceable(4000) %a.addr.0)
  call void @deref_phi_user(i32* %a.addr.0)
  %tmp = load i32, i32* %a.addr.0, align 4
  %cmp = icmp slt i32 %i.0, %tmp
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %incdec.ptr = getelementptr inbounds i32, i32* %a.addr.0, i64 -1
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}

; TEST 6
; loop in which dereferenceabily "shrinks"
define void @deref_phi_shrinking(i32* dereferenceable(4000) %a) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %a.addr.0 = phi i32* [ %a, %entry ], [ %incdec.ptr, %for.inc ]
; CHECK: call void @deref_phi_user(i32* %a.addr.0)
  call void @deref_phi_user(i32* %a.addr.0)
  %tmp = load i32, i32* %a.addr.0, align 4
  %cmp = icmp slt i32 %i.0, %tmp
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %incdec.ptr = getelementptr inbounds i32, i32* %a.addr.0, i64 1
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}
