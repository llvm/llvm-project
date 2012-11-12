; RUN: llc -march=x86-64 -mcpu=corei7 -disable-cgp-select2branch < %s

; We should not crash on this test.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.0.0"

@global = external constant [411 x i8], align 1

define void @snork() nounwind {
bb:
  br i1 undef, label %bb26, label %bb27

bb26:                                             ; preds = %bb48, %bb26, %bb
  switch i32 undef, label %bb26 [
    i32 142771596, label %bb28
  ]

bb27:                                             ; preds = %bb48, %bb
  switch i32 undef, label %bb49 [
    i32 142771596, label %bb28
  ]

bb28:                                             ; preds = %bb27, %bb26
  %tmp = load i32* null
  %tmp29 = trunc i32 %tmp to i8
  store i8* undef, i8** undef
  %tmp30 = load i32* null
  %tmp31 = icmp eq i32 %tmp30, 0
  %tmp32 = getelementptr inbounds [411 x i8]* @global, i32 0, i32 undef
  %tmp33 = load i8* %tmp32, align 1
  %tmp34 = getelementptr inbounds [411 x i8]* @global, i32 0, i32 0
  %tmp35 = load i8* %tmp34, align 1
  %tmp36 = select i1 %tmp31, i8 %tmp35, i8 %tmp33
  %tmp37 = select i1 undef, i8 %tmp29, i8 %tmp36
  %tmp38 = zext i8 %tmp37 to i32
  %tmp39 = select i1 undef, i32 0, i32 %tmp38
  %tmp40 = getelementptr inbounds i32* null, i32 %tmp39
  %tmp41 = load i32* %tmp40, align 4
  %tmp42 = load i32* undef, align 4
  %tmp43 = load i32* undef
  %tmp44 = xor i32 %tmp42, %tmp43
  %tmp45 = lshr i32 %tmp44, 8
  %tmp46 = lshr i32 %tmp44, 7
  call void @spam()
  unreachable

bb47:                                             ; No predecessors!
  ret void

bb48:                                             ; No predecessors!
  br i1 undef, label %bb27, label %bb26

bb49:                                             ; preds = %bb49, %bb27
  br label %bb49

bb50:                                             ; preds = %bb50
  br label %bb50
}

declare void @spam() noreturn nounwind
