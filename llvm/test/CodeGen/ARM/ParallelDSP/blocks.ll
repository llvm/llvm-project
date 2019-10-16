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

; TODO: Four smlads should be generated here, but mul.0 and mul.3 remain as
; scalars.
; CHECK-LABEL: num_load_limit
; CHECK: call i32 @llvm.arm.smlad
; CHECK: call i32 @llvm.arm.smlad
; CHECK: call i32 @llvm.arm.smlad
; CHECK-NOT: call i32 @llvm.arm.smlad
define i32 @num_load_limit(i16* %a, i16* %b, i32 %acc) {
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
  %add.0 = add i32 %mul.0, %mul.1

  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %ld.a.2 = load i16, i16* %addr.a.2
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %ld.b.2 = load i16, i16* %addr.b.2
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %mul.2 = mul i32 %sext.a.0, %sext.b.0
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %addr.b.3 = getelementptr i16, i16* %b, i32 3
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %ld.b.3 = load i16, i16* %addr.b.3
  %sext.b.3 = sext i16 %ld.b.3 to i32
  %mul.3 = mul i32 %sext.a.1, %sext.b.3
  %add.3 = add i32 %mul.2, %mul.3

  %addr.a.4 = getelementptr i16, i16* %a, i32 4
  %addr.b.4 = getelementptr i16, i16* %b, i32 4
  %ld.a.4 = load i16, i16* %addr.a.4
  %sext.a.4 = sext i16 %ld.a.4 to i32
  %ld.b.4 = load i16, i16* %addr.b.4
  %sext.b.4 = sext i16 %ld.b.4 to i32
  %mul.4 = mul i32 %sext.a.4, %sext.b.4
  %addr.a.5 = getelementptr i16, i16* %a, i32 5
  %addr.b.5 = getelementptr i16, i16* %b, i32 5
  %ld.a.5 = load i16, i16* %addr.a.5
  %sext.a.5 = sext i16 %ld.a.5 to i32
  %ld.b.5 = load i16, i16* %addr.b.5
  %sext.b.5 = sext i16 %ld.b.5 to i32
  %mul.5 = mul i32 %sext.a.5, %sext.b.5
  %add.5 = add i32 %mul.4, %mul.5

  %addr.a.6 = getelementptr i16, i16* %a, i32 6
  %addr.b.6 = getelementptr i16, i16* %b, i32 6
  %ld.a.6 = load i16, i16* %addr.a.6
  %sext.a.6 = sext i16 %ld.a.6 to i32
  %ld.b.6 = load i16, i16* %addr.b.6
  %sext.b.6 = sext i16 %ld.b.6 to i32
  %mul.6 = mul i32 %sext.a.6, %sext.b.6
  %addr.a.7 = getelementptr i16, i16* %a, i32 7
  %addr.b.7 = getelementptr i16, i16* %b, i32 7
  %ld.a.7 = load i16, i16* %addr.a.7
  %sext.a.7 = sext i16 %ld.a.7 to i32
  %ld.b.7 = load i16, i16* %addr.b.7
  %sext.b.7 = sext i16 %ld.b.7 to i32
  %mul.7 = mul i32 %sext.a.7, %sext.b.7
  %add.7 = add i32 %mul.6, %mul.7

  %add.10 = add i32 %add.7, %add.5
  %add.11 = add i32 %add.3, %add.0
  %add.12 = add i32 %add.10, %add.11
  %res = add i32 %add.12, %acc
  ret i32 %res
}

; CHECK-LABEL: too_many_loads
; CHECK-NOT: call i32 @llvm.arm.smlad
define i32 @too_many_loads(i16* %a, i16* %b, i32 %acc) {
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
  %add.0 = add i32 %mul.0, %mul.1

  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %ld.a.2 = load i16, i16* %addr.a.2
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %ld.b.2 = load i16, i16* %addr.b.2
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %mul.2 = mul i32 %sext.a.0, %sext.b.0
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %addr.b.3 = getelementptr i16, i16* %b, i32 3
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %ld.b.3 = load i16, i16* %addr.b.3
  %sext.b.3 = sext i16 %ld.b.3 to i32
  %mul.3 = mul i32 %sext.a.1, %sext.b.3
  %add.3 = add i32 %mul.2, %mul.3

  %addr.a.4 = getelementptr i16, i16* %a, i32 4
  %addr.b.4 = getelementptr i16, i16* %b, i32 4
  %ld.a.4 = load i16, i16* %addr.a.4
  %sext.a.4 = sext i16 %ld.a.4 to i32
  %ld.b.4 = load i16, i16* %addr.b.4
  %sext.b.4 = sext i16 %ld.b.4 to i32
  %mul.4 = mul i32 %sext.a.4, %sext.b.4
  %addr.a.5 = getelementptr i16, i16* %a, i32 5
  %addr.b.5 = getelementptr i16, i16* %b, i32 5
  %ld.a.5 = load i16, i16* %addr.a.5
  %sext.a.5 = sext i16 %ld.a.5 to i32
  %ld.b.5 = load i16, i16* %addr.b.5
  %sext.b.5 = sext i16 %ld.b.5 to i32
  %mul.5 = mul i32 %sext.a.5, %sext.b.5
  %add.5 = add i32 %mul.4, %mul.5

  %addr.a.6 = getelementptr i16, i16* %a, i32 6
  %addr.b.6 = getelementptr i16, i16* %b, i32 6
  %ld.a.6 = load i16, i16* %addr.a.6
  %sext.a.6 = sext i16 %ld.a.6 to i32
  %ld.b.6 = load i16, i16* %addr.b.6
  %sext.b.6 = sext i16 %ld.b.6 to i32
  %mul.6 = mul i32 %sext.a.6, %sext.b.6
  %addr.a.7 = getelementptr i16, i16* %a, i32 7
  %addr.b.7 = getelementptr i16, i16* %b, i32 7
  %ld.a.7 = load i16, i16* %addr.a.7
  %sext.a.7 = sext i16 %ld.a.7 to i32
  %ld.b.7 = load i16, i16* %addr.b.7
  %sext.b.7 = sext i16 %ld.b.7 to i32
  %mul.7 = mul i32 %sext.a.7, %sext.b.7
  %add.7 = add i32 %mul.6, %mul.7

  %addr.a.8 = getelementptr i16, i16* %a, i32 7
  %addr.b.8 = getelementptr i16, i16* %b, i32 7
  %ld.a.8 = load i16, i16* %addr.a.8
  %sext.a.8 = sext i16 %ld.a.8 to i32
  %ld.b.8 = load i16, i16* %addr.b.8
  %sext.b.8 = sext i16 %ld.b.8 to i32
  %mul.8 = mul i32 %sext.a.8, %sext.b.8

  %add.10 = add i32 %add.7, %add.5
  %add.11 = add i32 %add.3, %add.0
  %add.12 = add i32 %add.10, %add.11
  %add.13 = add i32 %add.12, %acc
  %res = add i32 %add.13, %mul.8
  ret i32 %res
}
