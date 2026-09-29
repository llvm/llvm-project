; RUN: opt -S -loop-reduce < %s | FileCheck %s

source_filename = "./simple.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: test
define void @test() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %tmp = phi i32 [ undef, %bb ], [ %tmp87, %bb1 ]
  %tmp2 = phi i32 [ undef, %bb ], [ %tmp86, %bb1 ]
  %tmp3 = mul i32 %tmp, undef
  %tmp4 = xor i32 %tmp3, -1
  %tmp5 = add i32 %tmp, %tmp4
  %tmp6 = add i32 %tmp2, -1
  %tmp7 = add i32 %tmp5, %tmp6
  %tmp8 = mul i32 %tmp7, %tmp3
  %tmp9 = xor i32 %tmp8, -1
  %tmp10 = add i32 %tmp7, %tmp9
  %tmp11 = add i32 %tmp10, undef
  %tmp12 = mul i32 %tmp11, %tmp8
  %tmp13 = xor i32 %tmp12, -1
  %tmp14 = add i32 %tmp11, %tmp13
  %tmp15 = add i32 %tmp14, undef
  %tmp16 = mul i32 %tmp15, %tmp12
  %tmp17 = add i32 %tmp15, undef
  %tmp18 = add i32 %tmp17, undef
  %tmp19 = mul i32 %tmp18, %tmp16
  %tmp20 = xor i32 %tmp19, -1
  %tmp21 = add i32 %tmp18, %tmp20
  %tmp22 = add i32 %tmp21, undef
  %tmp23 = mul i32 %tmp22, %tmp19
  %tmp24 = xor i32 %tmp23, -1
  %tmp25 = add i32 %tmp22, %tmp24
  %tmp26 = add i32 %tmp25, undef
  %tmp27 = mul i32 %tmp26, %tmp23
  %tmp28 = xor i32 %tmp27, -1
  %tmp29 = add i32 %tmp26, %tmp28
  %tmp30 = add i32 %tmp29, undef
  %tmp31 = mul i32 %tmp30, %tmp27
  %tmp32 = xor i32 %tmp31, -1
  %tmp33 = add i32 %tmp30, %tmp32
  %tmp34 = add i32 %tmp33, undef
  %tmp35 = mul i32 %tmp34, %tmp31
  %tmp36 = xor i32 %tmp35, -1
  %tmp37 = add i32 %tmp34, %tmp36
  %tmp38 = add i32 %tmp2, -9
  %tmp39 = add i32 %tmp37, %tmp38
  %tmp40 = mul i32 %tmp39, %tmp35
  %tmp41 = xor i32 %tmp40, -1
  %tmp42 = add i32 %tmp39, %tmp41
  %tmp43 = add i32 %tmp42, undef
  %tmp44 = mul i32 %tmp43, %tmp40
  %tmp45 = xor i32 %tmp44, -1
  %tmp46 = add i32 %tmp43, %tmp45
  %tmp47 = add i32 %tmp46, undef
  %tmp48 = mul i32 %tmp47, %tmp44
  %tmp49 = xor i32 %tmp48, -1
  %tmp50 = add i32 %tmp47, %tmp49
  %tmp51 = add i32 %tmp50, undef
  %tmp52 = mul i32 %tmp51, %tmp48
  %tmp53 = xor i32 %tmp52, -1
  %tmp54 = add i32 %tmp51, %tmp53
  %tmp55 = add i32 %tmp54, undef
  %tmp56 = mul i32 %tmp55, %tmp52
  %tmp57 = xor i32 %tmp56, -1
  %tmp58 = add i32 %tmp55, %tmp57
  %tmp59 = add i32 %tmp2, -14
  %tmp60 = add i32 %tmp58, %tmp59
  %tmp61 = mul i32 %tmp60, %tmp56
  %tmp62 = xor i32 %tmp61, -1
  %tmp63 = add i32 %tmp60, %tmp62
  %tmp64 = add i32 %tmp63, undef
  %tmp65 = mul i32 %tmp64, %tmp61
  %tmp66 = xor i32 %tmp65, -1
  %tmp67 = add i32 %tmp64, %tmp66
  %tmp68 = add i32 %tmp67, undef
  %tmp69 = mul i32 %tmp68, %tmp65
  %tmp70 = xor i32 %tmp69, -1
  %tmp71 = add i32 %tmp68, %tmp70
  %tmp72 = add i32 %tmp71, undef
  %tmp73 = mul i32 %tmp72, %tmp69
  %tmp74 = xor i32 %tmp73, -1
  %tmp75 = add i32 %tmp72, %tmp74
  %tmp76 = add i32 %tmp75, undef
  %tmp77 = mul i32 %tmp76, %tmp73
  %tmp78 = xor i32 %tmp77, -1
  %tmp79 = add i32 %tmp76, %tmp78
  %tmp80 = add i32 %tmp79, undef
  %tmp81 = mul i32 %tmp80, %tmp77
  %tmp82 = xor i32 %tmp81, -1
  %tmp83 = add i32 %tmp80, %tmp82
  %tmp84 = add i32 %tmp83, undef
  %tmp85 = add i32 %tmp84, undef
  %tmp86 = add i32 %tmp2, -21
  %tmp87 = add i32 %tmp85, %tmp86
  br label %bb1
}
