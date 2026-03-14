; RUN: opt < %s -passes=indvars -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This test takes excessively long time if SCEV tries to construct huge
; SCEVMulExpr's (with ~1000 ops) due to non-linear analysis cost.
define i32 @test() {
; CHECK-LABEL: @test(
bci_0:
  br label %bci_12

bci_133:                                          ; preds = %bci_127.unr-lcssa
  ret i32 %tmp17

bci_12:                                           ; preds = %bci_127.unr-lcssa, %bci_0
  %indvars.iv184 = phi i64 [ %indvars.iv.next185, %bci_127.unr-lcssa ], [ 3, %bci_0 ]
  %tmp1 = trunc i64 %indvars.iv184 to i32
  br label %bci_55.postloop

bci_127.unr-lcssa:                                ; preds = %bci_90.postloop
  %indvars.iv.next185 = add nuw nsw i64 %indvars.iv184, 1
  %tmp4 = icmp sgt i64 %indvars.iv184, 91
  br i1 %tmp4, label %bci_133, label %bci_12

bci_55.postloop:                                  ; preds = %bci_90.postloop, %bci_12
  %indvars.iv180.postloop = phi i64 [ %indvars.iv.next181.postloop, %bci_90.postloop ], [ 15, %bci_12 ]
  %local_2_16.postloop = phi i32 [ %tmp17, %bci_90.postloop ], [ 4, %bci_12 ]
  %indvars.iv.next181.postloop = add nuw nsw i64 %indvars.iv180.postloop, 1
  %tmp6 = load i32, ptr addrspace(1) undef, align 4
  %tmp7 = mul i32 %tmp6, %tmp1
  br label %not_zero65.us.postloop

not_zero65.us.postloop:                           ; preds = %not_zero65.us.postloop.1, %bci_55.postloop
  %local_2_24.us.postloop = phi i32 [ %local_2_16.postloop, %bci_55.postloop ], [ %tmp49, %not_zero65.us.postloop.1 ]
  %local_6_.us.postloop = phi i32 [ 3, %bci_55.postloop ], [ %tmp50, %not_zero65.us.postloop.1 ]
  %tmp8 = mul i32 %tmp7, %local_2_24.us.postloop
  %tmp9 = mul i32 %tmp8, %local_2_24.us.postloop
  %tmp10 = mul i32 %tmp7, %tmp9
  %tmp11 = mul i32 %tmp10, %tmp9
  %tmp12 = mul i32 %tmp7, %tmp11
  %tmp13 = mul i32 %tmp12, %tmp11
  %tmp14 = mul i32 %tmp7, %tmp13
  %tmp15 = mul i32 %tmp14, %tmp13
  %tmp16 = mul i32 %tmp7, %tmp15
  %tmp17 = mul i32 %tmp16, %tmp15
  %tmp18 = icmp sgt i32 %local_6_.us.postloop, 82
  br i1 %tmp18, label %bci_90.postloop, label %not_zero65.us.postloop.1

bci_90.postloop:                                  ; preds = %not_zero65.us.postloop
  %tmp19 = icmp sgt i64 %indvars.iv180.postloop, 68
  br i1 %tmp19, label %bci_127.unr-lcssa, label %bci_55.postloop

not_zero65.us.postloop.1:                         ; preds = %not_zero65.us.postloop
  %tmp20 = mul i32 %tmp7, %tmp17
  %tmp21 = mul i32 %tmp20, %tmp17
  %tmp22 = mul i32 %tmp7, %tmp21
  %tmp23 = mul i32 %tmp22, %tmp21
  %tmp24 = mul i32 %tmp7, %tmp23
  %tmp25 = mul i32 %tmp24, %tmp23
  %tmp26 = mul i32 %tmp7, %tmp25
  %tmp27 = mul i32 %tmp26, %tmp25
  %tmp28 = mul i32 %tmp7, %tmp27
  %tmp29 = mul i32 %tmp28, %tmp27
  %tmp30 = mul i32 %tmp7, %tmp29
  %tmp31 = mul i32 %tmp30, %tmp29
  %tmp32 = mul i32 %tmp7, %tmp31
  %tmp33 = mul i32 %tmp32, %tmp31
  %tmp34 = mul i32 %tmp7, %tmp33
  %tmp35 = mul i32 %tmp34, %tmp33
  %tmp36 = mul i32 %tmp7, %tmp35
  %tmp37 = mul i32 %tmp36, %tmp35
  %tmp38 = mul i32 %tmp7, %tmp37
  %tmp39 = mul i32 %tmp38, %tmp37
  %tmp40 = mul i32 %tmp7, %tmp39
  %tmp41 = mul i32 %tmp40, %tmp39
  %tmp42 = mul i32 %tmp7, %tmp41
  %tmp43 = mul i32 %tmp42, %tmp41
  %tmp44 = mul i32 %tmp7, %tmp43
  %tmp45 = mul i32 %tmp44, %tmp43
  %tmp46 = mul i32 %tmp7, %tmp45
  %tmp47 = mul i32 %tmp46, %tmp45
  %tmp48 = mul i32 %tmp7, %tmp47
  %tmp49 = mul i32 %tmp48, %tmp47
  %tmp50 = add nsw i32 %local_6_.us.postloop, 20
  br label %not_zero65.us.postloop
}
