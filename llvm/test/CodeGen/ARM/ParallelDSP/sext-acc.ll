; RUN: opt -arm-parallel-dsp -dce -mtriple=armv7-a -S %s -o - | FileCheck %s

; CHECK-LABEL: sext_acc_1
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[ACC:%[^ ]+]] = sext i32 %acc to i64
; CHECK: call i64 @llvm.arm.smlald(i32 [[A]], i32 [[B]], i64 [[ACC]])
define i64 @sext_acc_1(i16* %a, i16* %b, i32 %acc) {
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
  %sext.acc = sext i32 %acc to i64
  %res = add i64 %add, %sext.acc
  ret i64 %res
}

; CHECK-LABEL: sext_acc_2
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* %addr.a.2 to i32*
; CHECK: [[A_2:%[^ ]+]] = load i32, i32* %4
; CHECK: [[CAST_B_2:%[^ ]+]] = bitcast i16* %addr.b.2 to i32*
; CHECK: [[B_2:%[^ ]+]] = load i32, i32* %6
; CHECK: [[ACC:%[^ ]+]] = sext i32 %acc to i64
; CHECK: [[SMLALD:%[^ ]+]] = call i64 @llvm.arm.smlald(i32 [[A]], i32 [[B]], i64 [[ACC]])
; CHECK: call i64 @llvm.arm.smlald(i32 [[A_2]], i32 [[B_2]], i64 [[SMLALD]])
define i64 @sext_acc_2(i16* %a, i16* %b, i32 %acc) {
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
  %sext.acc = sext i32 %acc to i64
  %add.1 = add i64 %add, %sext.acc
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %ld.a.2 = load i16, i16* %addr.a.2
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %ld.b.2 = load i16, i16* %addr.b.2
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %mul.2 = mul i32 %sext.a.2, %sext.b.2
  %sext.mul.2 = sext i32 %mul.2 to i64
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %addr.b.3 = getelementptr i16, i16* %b, i32 3
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %ld.b.3 = load i16, i16* %addr.b.3
  %sext.b.3 = sext i16 %ld.b.3 to i32
  %mul.3 = mul i32 %sext.a.3, %sext.b.3
  %sext.mul.3 = sext i32 %mul.3 to i64
  %add.2 = add i64 %sext.mul.2, %sext.mul.3
  %add.3 = add i64 %add.1, %add.2
  ret i64 %add.3
}

; CHECK-LABEL: sext_acc_3
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* %addr.a.2 to i32*
; CHECK: [[A_2:%[^ ]+]] = load i32, i32* %4
; CHECK: [[CAST_B_2:%[^ ]+]] = bitcast i16* %addr.b.2 to i32*
; CHECK: [[B_2:%[^ ]+]] = load i32, i32* %6
; CHECK: [[ACC:%[^ ]+]] = sext i32 %acc to i64
; CHECK: [[SMLALD:%[^ ]+]] = call i64 @llvm.arm.smlald(i32 [[A]], i32 [[B]], i64 [[ACC]])
; CHECK: call i64 @llvm.arm.smlald(i32 [[A_2]], i32 [[B_2]], i64 [[SMLALD]])
define i64 @sext_acc_3(i16* %a, i16* %b, i32 %acc) {
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
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %ld.a.2 = load i16, i16* %addr.a.2
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %ld.b.2 = load i16, i16* %addr.b.2
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %mul.2 = mul i32 %sext.a.2, %sext.b.2
  %sext.mul.2 = sext i32 %mul.2 to i64
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %addr.b.3 = getelementptr i16, i16* %b, i32 3
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %ld.b.3 = load i16, i16* %addr.b.3
  %sext.b.3 = sext i16 %ld.b.3 to i32
  %mul.3 = mul i32 %sext.a.3, %sext.b.3
  %sext.mul.3 = sext i32 %mul.3 to i64
  %add.1 = add i64 %sext.mul.2, %sext.mul.3
  %add.2 = add i64 %add, %add.1
  %sext.acc = sext i32 %acc to i64
  %add.3 = add i64 %add.2, %sext.acc
  ret i64 %add.3
}

; CHECK-LABEL: sext_acc_4
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* %addr.a.2 to i32*
; CHECK: [[A_2:%[^ ]+]] = load i32, i32* %4
; CHECK: [[CAST_B_2:%[^ ]+]] = bitcast i16* %addr.b.2 to i32*
; CHECK: [[B_2:%[^ ]+]] = load i32, i32* %6
; CHECK: [[ACC:%[^ ]+]] = sext i32 %acc to i64
; CHECK: [[SMLALD:%[^ ]+]] = call i64 @llvm.arm.smlald(i32 [[A]], i32 [[B]], i64 [[ACC]])
; CHECK: call i64 @llvm.arm.smlald(i32 [[A_2]], i32 [[B_2]], i64 [[SMLALD]])
define i64 @sext_acc_4(i16* %a, i16* %b, i32 %acc) {
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
  %sext.add = sext i32 %add to i64
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %ld.a.2 = load i16, i16* %addr.a.2
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %ld.b.2 = load i16, i16* %addr.b.2
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %mul.2 = mul i32 %sext.a.2, %sext.b.2
  %sext.mul.2 = sext i32 %mul.2 to i64
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %addr.b.3 = getelementptr i16, i16* %b, i32 3
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %ld.b.3 = load i16, i16* %addr.b.3
  %sext.b.3 = sext i16 %ld.b.3 to i32
  %mul.3 = mul i32 %sext.a.3, %sext.b.3
  %sext.mul.3 = sext i32 %mul.3 to i64
  %sext.acc = sext i32 %acc to i64
  %add.1 = add i64 %sext.mul.2, %sext.add
  %add.2 = add i64 %sext.add, %add.1
  %add.3 = add i64 %add.2, %sext.mul.3
  %add.4 = add i64 %add.3, %sext.acc
  ret i64 %add.4
}
