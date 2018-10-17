; RUN: opt -mtriple=thumbv8m.main -mcpu=cortex-m33 -S -arm-parallel-dsp %s -o - | FileCheck %s
; RUN: opt -mtriple=thumbv7a-linux-android -arm-parallel-dsp -S %s -o - | FileCheck %s

; CHECK-LABEL: sext_multi_use_undef
define void @sext_multi_use_undef() {
entry:
  br label %for.body

for.body:
  %0 = load i16, i16* undef, align 2
  %conv3 = sext i16 %0 to i32
  %1 = load i16, i16* undef, align 2
  %conv7 = sext i16 %1 to i32
  %mul8 = mul nsw i32 %conv7, %conv3
  %x.addr.180 = getelementptr inbounds i16, i16* undef, i32 1
  %2 = load i16, i16* %x.addr.180, align 2
  %conv1582 = sext i16 %2 to i32
  %mul.i7284 = mul nsw i32 %conv7, %conv1582
  br label %for.body
}

; CHECK-LABEL: sext_multi_use
; CHECK: [[PtrA:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[DataA:%[^ ]+]] = load i32, i32* [[PtrA]], align 2
; CHECK: [[Top:%[^ ]+]] = ashr i32 [[DataA]], 16
; CHECK: [[Shl:%[^ ]+]] = shl i32 [[DataA]], 16
; CHECK: [[Bottom:%[^ ]+]] = ashr i32 [[Shl]], 16
; CHECK: [[DataB:%[^ ]+]] = load i16, i16* %b, align 2
; CHECK: [[SextB:%[^ ]+]] = sext i16 [[DataB]] to i32
; CHECK: [[Mul0:%[^ ]+]] = mul nsw i32 [[SextB]], [[Bottom]]
; CHECK: [[Mul1:%[^ ]+]] = mul nsw i32 [[SextB]], [[Top]]
define void @sext_multi_use(i16* %a, i16* %b) {
entry:
  br label %for.body

for.body:
  %0 = load i16, i16* %a, align 2
  %conv3 = sext i16 %0 to i32
  %1 = load i16, i16* %b, align 2
  %conv7 = sext i16 %1 to i32
  %mul8 = mul nsw i32 %conv7, %conv3
  %x.addr.180 = getelementptr inbounds i16, i16* %a, i32 1
  %2 = load i16, i16* %x.addr.180, align 2
  %conv1582 = sext i16 %2 to i32
  %mul.i7284 = mul nsw i32 %conv7, %conv1582
  br label %for.body
}

; CHECK-LABEL: sext_multi_use_reorder
; CHECK: [[PtrA:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[DataA:%[^ ]+]] = load i32, i32* [[PtrA]], align 2
; CHECK: [[Top:%[^ ]+]] = ashr i32 [[DataA]], 16
; CHECK: [[Shl:%[^ ]+]] = shl i32 [[DataA]], 16
; CHECK: [[Bottom:%[^ ]+]] = ashr i32 [[Shl]], 16
; CHECK: [[Mul0:%[^ ]+]] = mul nsw i32 [[Top]], [[Bottom]]
; CHECK: [[DataB:%[^ ]+]] = load i16, i16* %b, align 2
; CHECK: [[SextB:%[^ ]+]] = sext i16 [[DataB]] to i32
; CHECK: [[Mul1:%[^ ]+]] = mul nsw i32 [[Top]], [[SextB]]
define void @sext_multi_use_reorder(i16* %a, i16* %b) {
entry:
  br label %for.body

for.body:
  %0 = load i16, i16* %a, align 2
  %conv3 = sext i16 %0 to i32
  %x.addr.180 = getelementptr inbounds i16, i16* %a, i32 1
  %1 = load i16, i16* %x.addr.180, align 2
  %conv7 = sext i16 %1 to i32
  %mul8 = mul nsw i32 %conv7, %conv3
  %2 = load i16, i16* %b, align 2
  %conv1582 = sext i16 %2 to i32
  %mul.i7284 = mul nsw i32 %conv7, %conv1582
  br label %for.body
}
