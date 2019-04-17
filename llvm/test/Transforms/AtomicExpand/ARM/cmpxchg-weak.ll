; RUN: opt -atomic-expand -codegen-opt-level=1 -S -mtriple=thumbv7s-apple-ios7.0 %s | FileCheck %s

define i32 @test_cmpxchg_seq_cst(i32* %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: @test_cmpxchg_seq_cst
; Intrinsic for "dmb ishst" is then expected
; CHECK:     br label %[[START:.*]]

; CHECK: [[START]]:
; CHECK:     [[LOADED:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* %addr)
; CHECK:     [[SHOULD_STORE:%.*]] = icmp eq i32 [[LOADED]], %desired
; CHECK:     br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK:     call void @llvm.arm.dmb(i32 10)
; CHECK:     br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK:     [[STREX:%.*]] = call i32 @llvm.arm.strex.p0i32(i32 %new, i32* %addr)
; CHECK:     [[SUCCESS:%.*]] = icmp eq i32 [[STREX]], 0
; CHECK:     br i1 [[SUCCESS]], label %[[SUCCESS_BB:.*]], label %[[FAILURE_BB:.*]]

; CHECK: [[SUCCESS_BB]]:
; CHECK:     call void @llvm.arm.dmb(i32 11)
; CHECK:     br label %[[END:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK:     call void @llvm.arm.clrex()
; CHECK:     br label %[[FAILURE_BB]]

; CHECK: [[FAILURE_BB]]:
; CHECK:     call void @llvm.arm.dmb(i32 11)
; CHECK:     br label %[[END]]

; CHECK: [[END]]:
; CHECK:     [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK:     ret i32 [[LOADED]]

  %pair = cmpxchg weak i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %oldval = extractvalue { i32, i1 } %pair, 0
  ret i32 %oldval
}

define i1 @test_cmpxchg_weak_fail(i32* %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: @test_cmpxchg_weak_fail
; CHECK:     br label %[[START:.*]]

; CHECK: [[START]]:
; CHECK:     [[LOADED:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* %addr)
; CHECK:     [[SHOULD_STORE:%.*]] = icmp eq i32 [[LOADED]], %desired
; CHECK:     br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK:     call void @llvm.arm.dmb(i32 10)
; CHECK:     br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK:     [[STREX:%.*]] = call i32 @llvm.arm.strex.p0i32(i32 %new, i32* %addr)
; CHECK:     [[SUCCESS:%.*]] = icmp eq i32 [[STREX]], 0
; CHECK:     br i1 [[SUCCESS]], label %[[SUCCESS_BB:.*]], label %[[FAILURE_BB:.*]]

; CHECK: [[SUCCESS_BB]]:
; CHECK:     call void @llvm.arm.dmb(i32 11)
; CHECK:     br label %[[END:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK:     call void @llvm.arm.clrex()
; CHECK:     br label %[[FAILURE_BB]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NOT: dmb
; CHECK:     br label %[[END]]

; CHECK: [[END]]:
; CHECK:     [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK:     ret i1 [[SUCCESS]]

  %pair = cmpxchg weak i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  %oldval = extractvalue { i32, i1 } %pair, 1
  ret i1 %oldval
}

define i32 @test_cmpxchg_monotonic(i32* %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: @test_cmpxchg_monotonic
; CHECK-NOT: dmb
; CHECK:     br label %[[START:.*]]

; CHECK: [[START]]:
; CHECK:     [[LOADED:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* %addr)
; CHECK:     [[SHOULD_STORE:%.*]] = icmp eq i32 [[LOADED]], %desired
; CHECK:     br i1 [[SHOULD_STORE]], label %[[TRY_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK:     [[STREX:%.*]] = call i32 @llvm.arm.strex.p0i32(i32 %new, i32* %addr)
; CHECK:     [[SUCCESS:%.*]] = icmp eq i32 [[STREX]], 0
; CHECK:     br i1 [[SUCCESS]], label %[[SUCCESS_BB:.*]], label %[[FAILURE_BB:.*]]

; CHECK: [[SUCCESS_BB]]:
; CHECK-NOT: dmb
; CHECK:     br label %[[END:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK:     call void @llvm.arm.clrex()
; CHECK:     br label %[[FAILURE_BB]]

; CHECK: [[FAILURE_BB]]:
; CHECK-NOT: dmb
; CHECK:     br label %[[END]]

; CHECK: [[END]]:
; CHECK:     [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK:     ret i32 [[LOADED]]

  %pair = cmpxchg weak i32* %addr, i32 %desired, i32 %new monotonic monotonic
  %oldval = extractvalue { i32, i1 } %pair, 0
  ret i32 %oldval
}

define i32 @test_cmpxchg_seq_cst_minsize(i32* %addr, i32 %desired, i32 %new) minsize {
; CHECK-LABEL: @test_cmpxchg_seq_cst_minsize
; CHECK:     br label %[[START:.*]]

; CHECK: [[START]]:
; CHECK:     [[LOADED:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* %addr)
; CHECK:     [[SHOULD_STORE:%.*]] = icmp eq i32 [[LOADED]], %desired
; CHECK:     br i1 [[SHOULD_STORE]], label %[[FENCED_STORE:.*]], label %[[NO_STORE_BB:.*]]

; CHECK: [[FENCED_STORE]]:
; CHECK:     call void @llvm.arm.dmb(i32 10)
; CHECK:     br label %[[TRY_STORE:.*]]

; CHECK: [[TRY_STORE]]:
; CHECK:     [[STREX:%.*]] = call i32 @llvm.arm.strex.p0i32(i32 %new, i32* %addr)
; CHECK:     [[SUCCESS:%.*]] = icmp eq i32 [[STREX]], 0
; CHECK:     br i1 [[SUCCESS]], label %[[SUCCESS_BB:.*]], label %[[FAILURE_BB:.*]]

; CHECK: [[SUCCESS_BB]]:
; CHECK:     call void @llvm.arm.dmb(i32 11)
; CHECK:     br label %[[END:.*]]

; CHECK: [[NO_STORE_BB]]:
; CHECK:     call void @llvm.arm.clrex()
; CHECK:     br label %[[FAILURE_BB]]

; CHECK: [[FAILURE_BB]]:
; CHECK:     call void @llvm.arm.dmb(i32 11)
; CHECK:     br label %[[END]]

; CHECK: [[END]]:
; CHECK:     [[SUCCESS:%.*]] = phi i1 [ true, %[[SUCCESS_BB]] ], [ false, %[[FAILURE_BB]] ]
; CHECK:     ret i32 [[LOADED]]

  %pair = cmpxchg weak i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %oldval = extractvalue { i32, i1 } %pair, 0
  ret i32 %oldval
}
