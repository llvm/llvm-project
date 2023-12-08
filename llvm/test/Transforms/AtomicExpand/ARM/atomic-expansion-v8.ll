; RUN: opt -S -o - -mtriple=armv8-linux-gnueabihf -atomic-expand %s -codegen-opt-level=1 | FileCheck %s

define i8 @test_atomic_xchg_i8(ptr %ptr, i8 %xchgend) {
; CHECK-LABEL: @test_atomic_xchg_i8
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i8) %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[NEWVAL32:%.*]] = zext i8 %xchgend to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0(i32 [[NEWVAL32]], ptr elementtype(i8) %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: fence
; CHECK: ret i8 [[OLDVAL]]
  %res = atomicrmw xchg ptr %ptr, i8 %xchgend monotonic
  ret i8 %res
}

define i16 @test_atomic_add_i16(ptr %ptr, i16 %addend) {
; CHECK-LABEL: @test_atomic_add_i16
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i16) %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i16
; CHECK: [[NEWVAL:%.*]] = add i16 [[OLDVAL]], %addend
; CHECK: [[NEWVAL32:%.*]] = zext i16 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.stlex.p0(i32 [[NEWVAL32]], ptr elementtype(i16) %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: fence
; CHECK: ret i16 [[OLDVAL]]
  %res = atomicrmw add ptr %ptr, i16 %addend seq_cst
  ret i16 %res
}

define i32 @test_atomic_sub_i32(ptr %ptr, i32 %subend) {
; CHECK-LABEL: @test_atomic_sub_i32
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i32) %ptr)
; CHECK: [[NEWVAL:%.*]] = sub i32 [[OLDVAL]], %subend
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0(i32 [[NEWVAL]], ptr elementtype(i32) %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: fence
; CHECK: ret i32 [[OLDVAL]]
  %res = atomicrmw sub ptr %ptr, i32 %subend acquire
  ret i32 %res
}

define i64 @test_atomic_or_i64(ptr %ptr, i64 %orend) {
; CHECK-LABEL: @test_atomic_or_i64
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[LOHI:%.*]] = call { i32, i32 } @llvm.arm.ldaexd(ptr %ptr)
; CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
; CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
; CHECK: [[LO64:%.*]] = zext i32 [[LO]] to i64
; CHECK: [[HI64_TMP:%.*]] = zext i32 [[HI]] to i64
; CHECK: [[HI64:%.*]] = shl i64 [[HI64_TMP]], 32
; CHECK: [[OLDVAL:%.*]] = or i64 [[LO64]], [[HI64]]
; CHECK: [[NEWVAL:%.*]] = or i64 [[OLDVAL]], %orend
; CHECK: [[NEWLO:%.*]] = trunc i64 [[NEWVAL]] to i32
; CHECK: [[NEWHI_TMP:%.*]] = lshr i64 [[NEWVAL]], 32
; CHECK: [[NEWHI:%.*]] = trunc i64 [[NEWHI_TMP]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.stlexd(i32 [[NEWLO]], i32 [[NEWHI]], ptr %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: fence
; CHECK: ret i64 [[OLDVAL]]
  %res = atomicrmw or ptr %ptr, i64 %orend seq_cst
  ret i64 %res
}

define i8 @test_cmpxchg_i8_seqcst_seqcst(ptr %ptr, i8 %desired, i8 %newval) {
; CHECK-LABEL: @test_cmpxchg_i8_seqcst_seqcst
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i8) %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 %1 to i8
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i8 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK-NEXT: br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[LOADED_TRYSTORE:%.*]] = phi i8 [ [[OLDVAL]], %[[FENCED_STORE]] ]
; CHECK: [[NEWVAL32:%.*]] = zext i8 %newval to i32
; CHECK: [[TRYAGAIN:%.*]] =  call i32 @llvm.arm.stlex.p0(i32 [[NEWVAL32]], ptr elementtype(i8) %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: [[LOADED_NOSTORE:%.*]] = phi i8 [ [[OLDVAL]], %[[LOOP]] ]
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NEXT: [[LOADED_FAILURE:%.*]] = phi i8 [ [[LOADED_NOSTORE]], %[[NO_STORE_BB]] ]
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[LOADED_EXIT:%.*]] = phi i8 [ [[LOADED_TRYSTORE]], %[[SUCCESS_BB]] ], [ [[LOADED_FAILURE]], %[[FAILURE_BB]] ]
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i8 [[LOADED_EXIT]]

  %pairold = cmpxchg ptr %ptr, i8 %desired, i8 %newval seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pairold, 0
  ret i8 %old
}

define i16 @test_cmpxchg_i16_seqcst_monotonic(ptr %ptr, i16 %desired, i16 %newval) {
; CHECK-LABEL: @test_cmpxchg_i16_seqcst_monotonic
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i16) %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 %1 to i16
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i16 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK-NEXT: br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[LOADED_TRYSTORE:%.*]] = phi i16 [ [[OLDVAL]], %[[FENCED_STORE]] ]
; CHECK: [[NEWVAL32:%.*]] = zext i16 %newval to i32
; CHECK: [[TRYAGAIN:%.*]] =  call i32 @llvm.arm.stlex.p0(i32 [[NEWVAL32]], ptr elementtype(i16) %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: fence
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; The PHI is not required.
; CHECK-NEXT: [[LOADED_NOSTORE:%.*]] = phi i16 [ [[OLDVAL]], %[[LOOP]] ]
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NEXT: [[LOADED_FAILURE:%.*]] = phi i16 [ [[LOADED_NOSTORE]], %[[NO_STORE_BB]] ]
; CHECK-NOT: fence
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[LOADED_EXIT:%.*]] = phi i16 [ [[LOADED_TRYSTORE]], %[[SUCCESS_BB]] ], [ [[LOADED_FAILURE]], %[[FAILURE_BB]] ]
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i16 [[LOADED_EXIT]]

  %pairold = cmpxchg ptr %ptr, i16 %desired, i16 %newval seq_cst monotonic
  %old = extractvalue { i16, i1 } %pairold, 0
  ret i16 %old
}

define i32 @test_cmpxchg_i32_acquire_acquire(ptr %ptr, i32 %desired, i32 %newval) {
; CHECK-LABEL: @test_cmpxchg_i32_acquire_acquire
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i32) %ptr)
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i32 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK-NEXT: br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[LOADED_TRYSTORE:%.*]] = phi i32 [ [[OLDVAL]], %[[FENCED_STORE]] ]
; CHECK: [[TRYAGAIN:%.*]] =  call i32 @llvm.arm.strex.p0(i32 %newval, ptr elementtype(i32) %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: [[LOADED_NOSTORE:%.*]] = phi i32 [ [[OLDVAL]], %[[LOOP]] ]
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NEXT: [[LOADED_FAILURE:%.*]] = phi i32 [ [[LOADED_NOSTORE]], %[[NO_STORE_BB]] ]
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[LOADED_EXIT:%.*]] = phi i32 [ [[LOADED_TRYSTORE]], %[[SUCCESS_BB]] ], [ [[LOADED_FAILURE]], %[[FAILURE_BB]] ]
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i32 [[LOADED_EXIT]]

  %pairold = cmpxchg ptr %ptr, i32 %desired, i32 %newval acquire acquire
  %old = extractvalue { i32, i1 } %pairold, 0
  ret i32 %old
}

define i64 @test_cmpxchg_i64_monotonic_monotonic(ptr %ptr, i64 %desired, i64 %newval) {
; CHECK-LABEL: @test_cmpxchg_i64_monotonic_monotonic
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[LOHI:%.*]] = call { i32, i32 } @llvm.arm.ldrexd(ptr %ptr)
; CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
; CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
; CHECK: [[LO64:%.*]] = zext i32 [[LO]] to i64
; CHECK: [[HI64_TMP:%.*]] = zext i32 [[HI]] to i64
; CHECK: [[HI64:%.*]] = shl i64 [[HI64_TMP]], 32
; CHECK: [[OLDVAL:%.*]] = or i64 [[LO64]], [[HI64]]
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i64 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK-NEXT: br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[LOADED_TRYSTORE:%.*]] = phi i64 [ [[OLDVAL]], %[[FENCED_STORE]] ]
; CHECK: [[NEWLO:%.*]] = trunc i64 %newval to i32
; CHECK: [[NEWHI_TMP:%.*]] = lshr i64 %newval, 32
; CHECK: [[NEWHI:%.*]] = trunc i64 [[NEWHI_TMP]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strexd(i32 [[NEWLO]], i32 [[NEWHI]], ptr %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: [[LOADED_NOSTORE:%.*]] = phi i64 [ [[OLDVAL]], %[[LOOP]] ]
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NEXT: [[LOADED_FAILURE:%.*]] = phi i64 [ [[LOADED_NOSTORE]], %[[NO_STORE_BB]] ]
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[LOADED_EXIT:%.*]] = phi i64 [ [[LOADED_TRYSTORE]], %[[SUCCESS_BB]] ], [ [[LOADED_FAILURE]], %[[FAILURE_BB]] ]
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i64 [[LOADED_EXIT]]

  %pairold = cmpxchg ptr %ptr, i64 %desired, i64 %newval monotonic monotonic
  %old = extractvalue { i64, i1 } %pairold, 0
  ret i64 %old
}
