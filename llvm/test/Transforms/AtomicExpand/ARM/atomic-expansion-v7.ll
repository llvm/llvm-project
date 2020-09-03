; RUN: opt -S -o - -mtriple=armv7-apple-ios7.0 -atomic-expand -codegen-opt-level=1 %s | FileCheck %s

define i8 @test_atomic_xchg_i8(i8* %ptr, i8 %xchgend) {
; CHECK-LABEL: @test_atomic_xchg_i8
; CHECK-NOT: dmb
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[NEWVAL32:%.*]] = zext i8 %xchgend to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: dmb
; CHECK: ret i8 [[OLDVAL]]
  %res = atomicrmw xchg i8* %ptr, i8 %xchgend monotonic
  ret i8 %res
}

define i16 @test_atomic_add_i16(i16* %ptr, i16 %addend) {
; CHECK-LABEL: @test_atomic_add_i16
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i16(i16* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i16
; CHECK: [[NEWVAL:%.*]] = add i16 [[OLDVAL]], %addend
; CHECK: [[NEWVAL32:%.*]] = zext i16 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i16(i32 [[NEWVAL32]], i16* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: ret i16 [[OLDVAL]]
  %res = atomicrmw add i16* %ptr, i16 %addend seq_cst
  ret i16 %res
}

define i32 @test_atomic_sub_i32(i32* %ptr, i32 %subend) {
; CHECK-LABEL: @test_atomic_sub_i32
; CHECK-NOT: dmb
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* %ptr)
; CHECK: [[NEWVAL:%.*]] = sub i32 [[OLDVAL]], %subend
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i32(i32 [[NEWVAL]], i32* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: ret i32 [[OLDVAL]]
  %res = atomicrmw sub i32* %ptr, i32 %subend acquire
  ret i32 %res
}

define i8 @test_atomic_and_i8(i8* %ptr, i8 %andend) {
; CHECK-LABEL: @test_atomic_and_i8
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[NEWVAL:%.*]] = and i8 [[OLDVAL]], %andend
; CHECK: [[NEWVAL32:%.*]] = zext i8 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK-NOT: dmb
; CHECK: ret i8 [[OLDVAL]]
  %res = atomicrmw and i8* %ptr, i8 %andend release
  ret i8 %res
}

define i16 @test_atomic_nand_i16(i16* %ptr, i16 %nandend) {
; CHECK-LABEL: @test_atomic_nand_i16
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i16(i16* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i16
; CHECK: [[NEWVAL_TMP:%.*]] = and i16 [[OLDVAL]], %nandend
; CHECK: [[NEWVAL:%.*]] = xor i16 [[NEWVAL_TMP]], -1
; CHECK: [[NEWVAL32:%.*]] = zext i16 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i16(i32 [[NEWVAL32]], i16* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: ret i16 [[OLDVAL]]
  %res = atomicrmw nand i16* %ptr, i16 %nandend seq_cst
  ret i16 %res
}

define i64 @test_atomic_or_i64(i64* %ptr, i64 %orend) {
; CHECK-LABEL: @test_atomic_or_i64
; CHECK: call void @llvm.arm.dmb(i32 11)
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
; CHECK: [[NEWVAL:%.*]] = or i64 [[OLDVAL]], %orend
; CHECK: [[NEWLO:%.*]] = trunc i64 [[NEWVAL]] to i32
; CHECK: [[NEWHI_TMP:%.*]] = lshr i64 [[NEWVAL]], 32
; CHECK: [[NEWHI:%.*]] = trunc i64 [[NEWHI_TMP]] to i32
; CHECK: [[PTR8:%.*]] = bitcast i64* %ptr to i8*
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strexd(i32 [[NEWLO]], i32 [[NEWHI]], i8* [[PTR8]])
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: ret i64 [[OLDVAL]]
  %res = atomicrmw or i64* %ptr, i64 %orend seq_cst
  ret i64 %res
}

define i8 @test_atomic_xor_i8(i8* %ptr, i8 %xorend) {
; CHECK-LABEL: @test_atomic_xor_i8
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[NEWVAL:%.*]] = xor i8 [[OLDVAL]], %xorend
; CHECK: [[NEWVAL32:%.*]] = zext i8 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: ret i8 [[OLDVAL]]
  %res = atomicrmw xor i8* %ptr, i8 %xorend seq_cst
  ret i8 %res
}

define i8 @test_atomic_max_i8(i8* %ptr, i8 %maxend) {
; CHECK-LABEL: @test_atomic_max_i8
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[WANT_OLD:%.*]] = icmp sgt i8 [[OLDVAL]], %maxend
; CHECK: [[NEWVAL:%.*]] = select i1 [[WANT_OLD]], i8 [[OLDVAL]], i8 %maxend
; CHECK: [[NEWVAL32:%.*]] = zext i8 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: ret i8 [[OLDVAL]]
  %res = atomicrmw max i8* %ptr, i8 %maxend seq_cst
  ret i8 %res
}

define i8 @test_atomic_min_i8(i8* %ptr, i8 %minend) {
; CHECK-LABEL: @test_atomic_min_i8
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[WANT_OLD:%.*]] = icmp sle i8 [[OLDVAL]], %minend
; CHECK: [[NEWVAL:%.*]] = select i1 [[WANT_OLD]], i8 [[OLDVAL]], i8 %minend
; CHECK: [[NEWVAL32:%.*]] = zext i8 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: ret i8 [[OLDVAL]]
  %res = atomicrmw min i8* %ptr, i8 %minend seq_cst
  ret i8 %res
}

define i8 @test_atomic_umax_i8(i8* %ptr, i8 %umaxend) {
; CHECK-LABEL: @test_atomic_umax_i8
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[WANT_OLD:%.*]] = icmp ugt i8 [[OLDVAL]], %umaxend
; CHECK: [[NEWVAL:%.*]] = select i1 [[WANT_OLD]], i8 [[OLDVAL]], i8 %umaxend
; CHECK: [[NEWVAL32:%.*]] = zext i8 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: ret i8 [[OLDVAL]]
  %res = atomicrmw umax i8* %ptr, i8 %umaxend seq_cst
  ret i8 %res
}

define i8 @test_atomic_umin_i8(i8* %ptr, i8 %uminend) {
; CHECK-LABEL: @test_atomic_umin_i8
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]
; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[WANT_OLD:%.*]] = icmp ule i8 [[OLDVAL]], %uminend
; CHECK: [[NEWVAL:%.*]] = select i1 [[WANT_OLD]], i8 [[OLDVAL]], i8 %uminend
; CHECK: [[NEWVAL32:%.*]] = zext i8 [[NEWVAL]] to i32
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp ne i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[LOOP]], label %[[END:.*]]
; CHECK: [[END]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: ret i8 [[OLDVAL]]
  %res = atomicrmw umin i8* %ptr, i8 %uminend seq_cst
  ret i8 %res
}

define i8 @test_cmpxchg_i8_seqcst_seqcst(i8* %ptr, i8 %desired, i8 %newval) {
; CHECK-LABEL: @test_cmpxchg_i8_seqcst_seqcst
; CHECK: br label %[[START:.*]]

; CHECK: [[START]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 [[OLDVAL32]] to i8
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i8 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[LOADED_LOOP:%.*]] = phi i8 [ [[OLDVAL]], %[[FENCED_STORE]] ], [ [[OLDVAL_LOOP:%.*]], %[[RELEASED_LOAD:.*]] ]
; CHECK: [[NEWVAL32:%.*]] = zext i8 %newval to i32
; CHECK: [[TRYAGAIN:%.*]] =  call i32 @llvm.arm.strex.p0i8(i32 [[NEWVAL32]], i8* %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[RELEASED_LOAD]]

; CHECK: [[RELEASED_LOAD]]:
; CHECK: [[OLDVAL32_LOOP:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %ptr)
; CHECK: [[OLDVAL_LOOP]] = trunc i32 [[OLDVAL32_LOOP]] to i8
; CHECK: [[SHOULD_STORE_LOOP:%.*]] = icmp eq i8 [[OLDVAL_LOOP]], %desired
; CHECK: br i1 [[SHOULD_STORE_LOOP]], label %[[LOOP]], label %[[NO_STORE_BB]]

; CHECK: [[SUCCESS_BB]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: [[LOADED_NO_STORE:%.*]] = phi i8 [ [[OLDVAL]], %[[START]] ], [ [[OLDVAL_LOOP]], %[[RELEASED_LOAD]] ]
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK: [[LOADED_FAILURE:%.*]] = phi i8 [ [[LOADED_NO_STORE]], %[[NO_STORE_BB]] ]
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[LOADED:%.*]] = phi i8 [ [[LOADED_LOOP]], %[[SUCCESS_BB]] ], [ [[LOADED_FAILURE]], %[[FAILURE_BB]] ]
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i8 [[LOADED]]

  %pairold = cmpxchg i8* %ptr, i8 %desired, i8 %newval seq_cst seq_cst
  %old = extractvalue { i8, i1 } %pairold, 0
  ret i8 %old
}

define i16 @test_cmpxchg_i16_seqcst_monotonic(i16* %ptr, i16 %desired, i16 %newval) {
; CHECK-LABEL: @test_cmpxchg_i16_seqcst_monotonic
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL32:%.*]] = call i32 @llvm.arm.ldrex.p0i16(i16* %ptr)
; CHECK: [[OLDVAL:%.*]] = trunc i32 %1 to i16
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i16 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[LOADED_LOOP:%.*]] = phi i16 [ [[OLDVAL]], %[[FENCED_STORE]] ], [ [[OLDVAL_LOOP:%.*]], %[[RELEASED_LOAD:.*]] ]
; CHECK: [[NEWVAL32:%.*]] = zext i16 %newval to i32
; CHECK: [[TRYAGAIN:%.*]] =  call i32 @llvm.arm.strex.p0i16(i32 [[NEWVAL32]], i16* %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[RELEASED_LOAD:.*]]

; CHECK: [[RELEASED_LOAD]]:
; CHECK: [[OLDVAL32_LOOP:%.*]] = call i32 @llvm.arm.ldrex.p0i16(i16* %ptr)
; CHECK: [[OLDVAL_LOOP]] = trunc i32 [[OLDVAL32_LOOP]] to i16
; CHECK: [[SHOULD_STORE_LOOP:%.*]] = icmp eq i16 [[OLDVAL_LOOP]], %desired
; CHECK: br i1 [[SHOULD_STORE_LOOP]], label %[[LOOP]], label %[[NO_STORE_BB]]

; CHECK: [[SUCCESS_BB]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: [[LOADED_NO_STORE:%.*]] = phi i16 [ [[OLDVAL]], %[[START]] ], [ [[OLDVAL_LOOP]], %[[RELEASED_LOAD]] ]
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NEXT: [[LOADED_FAILURE:%.*]] = phi i16 [ [[LOADED_NO_STORE]], %[[NO_STORE_BB]] ]
; CHECK-NOT: dmb
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[LOADED:%.*]] = phi i16 [ [[LOADED_LOOP]], %[[SUCCESS_BB]] ], [ [[LOADED_FAILURE]], %[[FAILURE_BB]] ]
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i16 [[LOADED]]

  %pairold = cmpxchg i16* %ptr, i16 %desired, i16 %newval seq_cst monotonic
  %old = extractvalue { i16, i1 } %pairold, 0
  ret i16 %old
}

define i32 @test_cmpxchg_i32_acquire_acquire(i32* %ptr, i32 %desired, i32 %newval) {
; CHECK-LABEL: @test_cmpxchg_i32_acquire_acquire
; CHECK-NOT: dmb
; CHECK: br label %[[LOOP:.*]]

; CHECK: [[LOOP]]:
; CHECK: [[OLDVAL:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* %ptr)
; CHECK: [[SHOULD_STORE:%.*]] = icmp eq i32 [[OLDVAL]], %desired
; CHECK: br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK-NEXT: br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[LOADED_TRYSTORE:%.*]] = phi i32 [ [[OLDVAL]], %[[FENCED_STORE]] ]
; CHECK: [[TRYAGAIN:%.*]] =  call i32 @llvm.arm.strex.p0i32(i32 %newval, i32* %ptr)
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: [[LOADED_NO_STORE:%.*]] = phi i32 [ [[OLDVAL]], %[[LOOP]] ]
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK: [[LOADED_FAILURE:%.*]] = phi i32 [ [[LOADED_NO_STORE]], %[[NO_STORE_BB]] ]
; CHECK: call void @llvm.arm.dmb(i32 11)
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[LOADED_EXIT:%.*]] = phi i32 [ [[LOADED_TRYSTORE]], %[[SUCCESS_BB]] ], [ [[LOADED_FAILURE]], %[[FAILURE_BB]] ]
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i32 [[LOADED_EXIT]]

  %pairold = cmpxchg i32* %ptr, i32 %desired, i32 %newval acquire acquire
  %old = extractvalue { i32, i1 } %pairold, 0
  ret i32 %old
}

define i64 @test_cmpxchg_i64_monotonic_monotonic(i64* %ptr, i64 %desired, i64 %newval) {
; CHECK-LABEL: @test_cmpxchg_i64_monotonic_monotonic
; CHECK-NOT: dmb
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
; CHECK: br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK-NEXT: br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK: [[LOADED_TRYSTORE:%.*]] = phi i64 [ [[OLDVAL]], %[[FENCED_STORE]] ]
; CHECK: [[NEWLO:%.*]] = trunc i64 %newval to i32
; CHECK: [[NEWHI_TMP:%.*]] = lshr i64 %newval, 32
; CHECK: [[NEWHI:%.*]] = trunc i64 [[NEWHI_TMP]] to i32
; CHECK: [[PTR8:%.*]] = bitcast i64* %ptr to i8*
; CHECK: [[TRYAGAIN:%.*]] = call i32 @llvm.arm.strexd(i32 [[NEWLO]], i32 [[NEWHI]], i8* [[PTR8]])
; CHECK: [[TST:%.*]] = icmp eq i32 [[TRYAGAIN]], 0
; CHECK: br i1 [[TST]], label %[[SUCCESS_BB:.*]], label %[[LOOP]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: dmb
; CHECK: br label %[[DONE:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK-NEXT: [[LOADED_NO_STORE:%.*]] = phi i64 [ [[OLDVAL]], %[[LOOP]] ]
; CHECK-NEXT: call void @llvm.arm.clrex()
; CHECK-NEXT: br label %[[FAILURE_BB:.*]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NEXT: [[LOADED_FAILURE:%.*]] = phi i64 [ [[LOADED_NO_STORE]], %[[NO_STORE_BB]] ]
; CHECK-NOT: dmb
; CHECK: br label %[[DONE]]

; CHECK: [[DONE]]:
; CHECK: [[LOADED_EXIT:%.*]] = phi i64 [ [[LOADED_TRYSTORE]], %[[SUCCESS_BB]] ], [ [[LOADED_FAILURE]], %[[FAILURE_BB]] ]
; CHECK: [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK: ret i64 [[LOADED_EXIT]]

  %pairold = cmpxchg i64* %ptr, i64 %desired, i64 %newval monotonic monotonic
  %old = extractvalue { i64, i1 } %pairold, 0
  ret i64 %old
}

define i32 @test_cmpxchg_minsize(i32* %addr, i32 %desired, i32 %new) minsize {
; CHECK-LABEL: @test_cmpxchg_minsize
; CHECK:     call void @llvm.arm.dmb(i32 11)
; CHECK:     br label %[[START:.*]]

; CHECK: [[START]]:
; CHECK:     [[LOADED:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* %addr)
; CHECK:     [[SHOULD_STORE:%.*]] = icmp eq i32 [[LOADED]], %desired
; CHECK:     br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK-NEXT: br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK:     [[LOADED_TRYSTORE:%.*]] = phi i32 [ [[LOADED]], %[[FENCED_STORE]] ]
; CHECK:     [[STREX:%.*]] = call i32 @llvm.arm.strex.p0i32(i32 %new, i32* %addr)
; CHECK:     [[SUCCESS:%.*]] = icmp eq i32 [[STREX]], 0
; CHECK:     br i1 [[SUCCESS]], label %[[SUCCESS_BB:.*]], label %[[START]]

; CHECK: [[SUCCESS_BB]]:
; CHECK:     call void @llvm.arm.dmb(i32 11)
; CHECK:     br label %[[END:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK:     [[LOADED_NO_STORE:%.*]] = phi i32 [ [[LOADED]], %[[START]] ]
; CHECK:     call void @llvm.arm.clrex()
; CHECK:     br label %[[FAILURE_BB]]

; CHECK: [[FAILURE_BB]]:
; CHECK:     [[LOADED_FAILURE:%.*]] = phi i32 [ [[LOADED_NO_STORE]], %[[NO_STORE_BB]] ]
; CHECK:     call void @llvm.arm.dmb(i32 11)
; CHECK:     br label %[[END]]

; CHECK: [[END]]:
; CHECK: [[LOADED_EXIT:%.*]] = phi i32 [ [[LOADED_TRYSTORE]], %[[SUCCESS_BB]] ], [ [[LOADED_FAILURE]], %[[FAILURE_BB]] ]
; CHECK:     [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK:     ret i32 [[LOADED_EXIT]]

  %pair = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %oldval = extractvalue { i32, i1 } %pair, 0
  ret i32 %oldval
}
