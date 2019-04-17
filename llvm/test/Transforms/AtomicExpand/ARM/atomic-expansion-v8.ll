; RUN: opt -S -o - -mtriple=armv8-linux-gnueabihf -atomic-expand %s -codegen-opt-level=1 | FileCheck %s

define i8 @test_atomic_xchg_i8(i8* %ptr, i8 %xchgend) {
; CHECK-LABEL: @test_atomic_xchg_i8
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[NEWVAL32:%.*]] = zext i8 %xchgend to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: fence
; CHECK: ret i8 [[OLDVAL]]
  %res = atomicrmw xchg i8* %ptr, i8 %xchgend monotonic
  ret i8 %res
}

define i16 @test_atomic_add_i16(i16* %ptr, i16 %addend) {
; CHECK-LABEL: @test_atomic_add_i16
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldaex.p0i16(i16* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i16
; CHECK: [[NEWVAL:%.*]] = add i16 [[OLDVAL]], %addend
; CHECK: [[NEWVAL32:%.*]] = zext i16 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.stlex.p0i16(i32 [[NEWVAL32]], i16* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: fence
; CHECK: ret i16 [[OLDVAL]]
  %res = atomicrmw add i16* %ptr, i16 %addend seq_cst
  ret i16 %res
}

define i32 @test_atomic_sub_i32(i32* %ptr, i32 %subend) {
; CHECK-LABEL: @test_atomic_sub_i32
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL:%.*]] = call i32 @llvm.arm.ldaex.p0i32(i32* %ptr)
; CHECK: [[NEWVAL:%.*]] = sub i32 [[OLDVAL]], %subend
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i32(i32 [[NEWVAL]], i32* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: fence
; CHECK: ret i32 [[OLDVAL]]
  %res = atomicrmw sub i32* %ptr, i32 %subend acquire
  ret i32 %res
}

define i64 @test_atomic_or_i64(i64* %ptr, i64 %orend) {
; CHECK-LABEL: @test_atomic_or_i64
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[PTR8:%.*]] = bitcast i64* %ptr to i8*
; CHECK: [[LOHI:%.*]] = call { i32, i32 } @llvm.arm.ldaexd(i8* [[PTR8]])
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
; CHECK: [[PTR8:%.*]] = bitcast i64* %ptr to i8*
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.stlexd(i32 [[NEWLO]], i32 [[NEWHI]], i8* [[PTR8]])
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: fence
; CHECK: ret i64 [[OLDVAL]]
  %res = atomicrmw or i64* %ptr, i64 %orend seq_cst
  ret i64 %res
}

define i8 @test_cmpxchg_i8_seqcst_seqcst(i8* %ptr, i8 %desired, i8 %newval) {
; CHECK-LABEL: @test_cmpxchg_i8_seqcst_seqcst
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldaex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 %1 to i8
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i8 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[TRY_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[NEWVAL32:%.*]] = zext i8 %newval to i32
; CHECK: [[TRYAGAIN:%.*]] =  call i32 @llvm.arm.stlex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i8 [[OLDVAL]]

  %pairold = cmpxchg i8* %ptr, i8 %desired, i8 %newval seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pairold, 0
  ret i8 %old
}

define i16 @test_cmpxchg_i16_seqcst_monotonic(i16* %ptr, i16 %desired, i16 %newval) {
; CHECK-LABEL: @test_cmpxchg_i16_seqcst_monotonic
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldaex.p0i16(i16* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 %1 to i16
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i16 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[TRY_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[NEWVAL32:%.*]] = zext i16 %newval to i32
; CHECK: [[TRYAGAIN:%.*]] =  call i32 @llvm.arm.stlex.p0i16(i32 [[NEWVAL32]], i16* %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: fence
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NOT: fence
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i16 [[OLDVAL]]

  %pairold = cmpxchg i16* %ptr, i16 %desired, i16 %newval seq_cst monotonic
  %old = extractvalue { i16, i1 } %pairold, 0
  ret i16 %old
}

define i32 @test_cmpxchg_i32_acquire_acquire(i32* %ptr, i32 %desired, i32 %newval) {
; CHECK-LABEL: @test_cmpxchg_i32_acquire_acquire
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL:%.*]] = call i32 @llvm.arm.ldaex.p0i32(i32* %ptr)
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i32 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[TRY_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[TRYAGAIN:%.*]] =  call i32 @llvm.arm.strex.p0i32(i32 %newval, i32* %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i32 [[OLDVAL]]

  %pairold = cmpxchg i32* %ptr, i32 %desired, i32 %newval acquire acquire
  %old = extractvalue { i32, i1 } %pairold, 0
  ret i32 %old
}

define i64 @test_cmpxchg_i64_monotonic_monotonic(i64* %ptr, i64 %desired, i64 %newval) {
; CHECK-LABEL: @test_cmpxchg_i64_monotonic_monotonic
; CHECK-NOT: fence
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[PTR8:%.*]] = bitcast i64* %ptr to i8*
; CHECK: [[LOHI:%.*]] = call { i32, i32 } @llvm.arm.ldrexd(i8* [[PTR8]])
; CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
; CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
; CHECK: [[LO64:%.*]] = zext i32 [[LO]] to i64
; CHECK: [[HI64_TMP:%.*]] = zext i32 [[HI]] to i64
; CHECK: [[HI64:%.*]] = shl i64 [[HI64_TMP]], 32
; CHECK: [[OLDVAL:%.*]] = or i64 [[LO64]], [[HI64]]
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i64 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[TRY_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[NEWLO:%.*]] = trunc i64 %newval to i32
; CHECK: [[NEWHI_TMP:%.*]] = lshr i64 %newval, 32
; CHECK: [[NEWHI:%.*]] = trunc i64 [[NEWHI_TMP]] to i32
; CHECK: [[PTR8:%.*]] = bitcast i64* %ptr to i8*
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strexd(i32 [[NEWLO]], i32 [[NEWHI]], i8* [[PTR8]])
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NOT: fence_cst
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i64 [[OLDVAL]]

  %pairold = cmpxchg i64* %ptr, i64 %desired, i64 %newval monotonic monotonic
  %old = extractvalue { i64, i1 } %pairold, 0
  ret i64 %old
}
