; RUN: opt -arm-parallel-dsp -dce -mtriple=armv7-a -S %s -o - | FileCheck %s

; CHECK-LABEL: single_block
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: call i32 @llvm.arm.smlad(i32 [[A]], i32 [[B]], i32 %acc)
define i32 @single_block(i16* %a, i16* %b, i32 %acc) {
entry:
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.1 = load i16, i16* %addr.a.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %add = add i32 %mul.0, %mul.1
  %res = add i32 %add, %acc
  ret i32 %res
}

; CHECK-LABEL: single_block_64
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: call i64 @llvm.arm.smlald(i32 [[A]], i32 [[B]], i64 %acc)
define i64 @single_block_64(i16* %a, i16* %b, i64 %acc) {
entry:
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.1 = load i16, i16* %addr.a.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %sext.mul.0 = sext i32 %mul.0 to i64
  %sext.mul.1 = sext i32 %mul.1 to i64
  %add = add i64 %sext.mul.0, %sext.mul.1
  %res = add i64 %add, %acc
  ret i64 %res
}

; CHECK-LABEL: multi_block
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK:  call i32 @llvm.arm.smlad(i32 [[A]], i32 [[B]], i32 0)
define i32 @multi_block(i16* %a, i16* %b, i32 %acc) {
entry:
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.1 = load i16, i16* %addr.a.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %add = add i32 %mul.0, %mul.1
  br label %bb.1

bb.1:
  %res = add i32 %add, %acc
  ret i32 %res
}

; CHECK-LABEL: multi_block_64
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK:  call i64 @llvm.arm.smlald(i32 [[A]], i32 [[B]], i64 0)
define i64 @multi_block_64(i16* %a, i16* %b, i64 %acc) {
entry:
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.1 = load i16, i16* %addr.a.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %sext.mul.0 = sext i32 %mul.0 to i64
  %sext.mul.1 = sext i32 %mul.1 to i64
  %add = add i64 %sext.mul.0, %sext.mul.1
  br label %bb.1

bb.1:
  %res = add i64 %add, %acc
  ret i64 %res
}

; CHECK-LABEL: multi_block_1
; CHECK-NOT: call i32 @llvm.arm.smlad
define i32 @multi_block_1(i16* %a, i16* %b, i32 %acc) {
entry:
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  br label %bb.1

bb.1:
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.1 = load i16, i16* %addr.a.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %add = add i32 %mul.0, %mul.1
  %res = add i32 %add, %acc
  ret i32 %res
}

