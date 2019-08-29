; RUN: opt -mtriple=thumbv7-unknown-linux-gnueabihf -arm-parallel-dsp -dce %s -S -o - | FileCheck %s

; CHECK-LABEL: first_mul_invalid
; CHECK: [[ADDR_IN_MINUS_1:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -1
; CHECK: [[LD_IN_MINUS_1:%[^ ]+]] = load i16, i16* [[ADDR_IN_MINUS_1]], align 2
; CHECK: [[IN_MINUS_1:%[^ ]+]] = sext i16 [[LD_IN_MINUS_1]] to i32
; CHECK: [[ADDR_B_PLUS_1:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 1
; CHECK: [[LD_B_PLUS_1:%[^ ]+]] = load i16, i16* [[ADDR_B_PLUS_1]], align 2
; CHECK: [[B_PLUS_1:%[^ ]+]] = sext i16 [[LD_B_PLUS_1]] to i32
; CHECK: [[MUL0:%[^ ]+]] = mul nsw i32 [[B_PLUS_1]], [[IN_MINUS_1]]
; CHECK: [[ADD0:%[^ ]+]] = add i32 [[MUL0]], %call
; CHECK: [[ADDR_IN_MINUS_3:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -3
; CHECK: [[CAST_ADDR_IN_MINUS_3:%[^ ]+]] = bitcast i16* [[ADDR_IN_MINUS_3]] to i32*
; CHECK: [[IN_MINUS_3:%[^ ]+]] = load i32, i32* [[CAST_ADDR_IN_MINUS_3]], align 2
; CHECK: [[ADDR_B_PLUS_2:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 2
; CHECK: [[CAST_ADDR_B_PLUS_2:%[^ ]+]] = bitcast i16* [[ADDR_B_PLUS_2]] to i32*
; CHECK: [[B_PLUS_2:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_PLUS_2]], align 2
; CHECK: [[ADDR_IN_MINUS_5:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -5
; CHECK: [[CAST_ADDR_IN_MINUS_5:%[^ ]+]] = bitcast i16* [[ADDR_IN_MINUS_5]] to i32*
; CHECK: [[IN_MINUS_5:%[^ ]+]] = load i32, i32* [[CAST_ADDR_IN_MINUS_5]], align 2
; CHECK: [[ADDR_B_PLUS_4:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 4
; CHECK: [[CAST_ADDR_B_PLUS_4:%[^ ]+]] = bitcast i16* [[ADDR_B_PLUS_4]] to i32*
; CHECK: [[B_PLUS_4:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_PLUS_4]], align 2
; CHECK: [[ACC:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN_MINUS_5]], i32 [[B_PLUS_4]], i32 [[ADD0]])
; CHECK: [[RES:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN_MINUS_3]], i32 [[B_PLUS_2]], i32 [[ACC]])
; CHECK: ret i32 [[RES]]
define i32 @first_mul_invalid(i16* nocapture readonly %in, i16* nocapture readonly %b) {
entry:
  %0 = load i16, i16* %in, align 2
  %conv = sext i16 %0 to i32
  %1 = load i16, i16* %b, align 2
  %conv2 = sext i16 %1 to i32
  %call = tail call i32 @bar(i32 %conv, i32 %conv2)
  %arrayidx3 = getelementptr inbounds i16, i16* %in, i32 -1
  %2 = load i16, i16* %arrayidx3, align 2
  %conv4 = sext i16 %2 to i32
  %arrayidx5 = getelementptr inbounds i16, i16* %b, i32 1
  %3 = load i16, i16* %arrayidx5, align 2
  %conv6 = sext i16 %3 to i32
  %mul = mul nsw i32 %conv6, %conv4
  %add = add i32 %mul, %call
  %arrayidx7 = getelementptr inbounds i16, i16* %in, i32 -2
  %4 = load i16, i16* %arrayidx7, align 2
  %conv8 = sext i16 %4 to i32
  %arrayidx9 = getelementptr inbounds i16, i16* %b, i32 2
  %5 = load i16, i16* %arrayidx9, align 2
  %conv10 = sext i16 %5 to i32
  %mul11 = mul nsw i32 %conv10, %conv8
  %add12 = add i32 %add, %mul11
  %arrayidx13 = getelementptr inbounds i16, i16* %in, i32 -3
  %6 = load i16, i16* %arrayidx13, align 2
  %conv14 = sext i16 %6 to i32
  %arrayidx15 = getelementptr inbounds i16, i16* %b, i32 3
  %7 = load i16, i16* %arrayidx15, align 2
  %conv16 = sext i16 %7 to i32
  %mul17 = mul nsw i32 %conv16, %conv14
  %add18 = add i32 %add12, %mul17
  %arrayidx19 = getelementptr inbounds i16, i16* %in, i32 -4
  %8 = load i16, i16* %arrayidx19, align 2
  %conv20 = sext i16 %8 to i32
  %arrayidx21 = getelementptr inbounds i16, i16* %b, i32 4
  %9 = load i16, i16* %arrayidx21, align 2
  %conv22 = sext i16 %9 to i32
  %mul23 = mul nsw i32 %conv22, %conv20
  %add24 = add i32 %add18, %mul23
  %arrayidx25 = getelementptr inbounds i16, i16* %in, i32 -5
  %10 = load i16, i16* %arrayidx25, align 2
  %conv26 = sext i16 %10 to i32
  %arrayidx27 = getelementptr inbounds i16, i16* %b, i32 5
  %11 = load i16, i16* %arrayidx27, align 2
  %conv28 = sext i16 %11 to i32
  %mul29 = mul nsw i32 %conv28, %conv26
  %add30 = add i32 %add24, %mul29
  ret i32 %add30
}

; CHECK-LABEL: with_no_acc_input
; CHECK: [[ADDR_IN_MINUS_1:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -1
; CHECK: [[LD_IN_MINUS_1:%[^ ]+]] = load i16, i16* [[ADDR_IN_MINUS_1]], align 2
; CHECK: [[IN_MINUS_1:%[^ ]+]] = sext i16 [[LD_IN_MINUS_1]] to i32
; CHECK: [[ADDR_B_PLUS_1:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 1
; CHECK: [[LD_B_PLUS_1:%[^ ]+]] = load i16, i16* [[ADDR_B_PLUS_1]], align 2
; CHECK: [[B_PLUS_1:%[^ ]+]] = sext i16 [[LD_B_PLUS_1]] to i32
; CHECK: [[MUL0:%[^ ]+]] = mul nsw i32 [[B_PLUS_1]], [[IN_MINUS_1]]
; CHECK: [[ADDR_IN_MINUS_3:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -3
; CHECK: [[CAST_ADDR_IN_MINUS_3:%[^ ]+]] = bitcast i16* [[ADDR_IN_MINUS_3]] to i32*
; CHECK: [[IN_MINUS_3:%[^ ]+]] = load i32, i32* [[CAST_ADDR_IN_MINUS_3]], align 2
; CHECK: [[ADDR_B_PLUS_2:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 2
; CHECK: [[CAST_ADDR_B_PLUS_2:%[^ ]+]] = bitcast i16* [[ADDR_B_PLUS_2]] to i32*
; CHECK: [[B_PLUS_2:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_PLUS_2]], align 2
; CHECK: [[ADDR_IN_MINUS_5:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -5
; CHECK: [[CAST_ADDR_IN_MINUS_5:%[^ ]+]] = bitcast i16* [[ADDR_IN_MINUS_5]] to i32*
; CHECK: [[IN_MINUS_5:%[^ ]+]] = load i32, i32* [[CAST_ADDR_IN_MINUS_5]], align 2
; CHECK: [[ADDR_B_PLUS_4:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 4
; CHECK: [[CAST_ADDR_B_PLUS_4:%[^ ]+]] = bitcast i16* [[ADDR_B_PLUS_4]] to i32*
; CHECK: [[B_PLUS_4:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_PLUS_4]], align 2
; CHECK: [[ACC:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN_MINUS_5]], i32 [[B_PLUS_4]], i32 [[MUL0]])
; CHECK: [[RES:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN_MINUS_3]], i32 [[B_PLUS_2]], i32 [[ACC]])
; CHECK: ret i32 [[RES]]
define i32 @with_no_acc_input(i16* nocapture readonly %in, i16* nocapture readonly %b) {
entry:
  %arrayidx3 = getelementptr inbounds i16, i16* %in, i32 -1
  %ld.2 = load i16, i16* %arrayidx3, align 2
  %conv4 = sext i16 %ld.2 to i32
  %arrayidx5 = getelementptr inbounds i16, i16* %b, i32 1
  %ld.3 = load i16, i16* %arrayidx5, align 2
  %conv6 = sext i16 %ld.3 to i32
  %mul = mul nsw i32 %conv6, %conv4
  %arrayidx7 = getelementptr inbounds i16, i16* %in, i32 -2
  %ld.4 = load i16, i16* %arrayidx7, align 2
  %conv8 = sext i16 %ld.4 to i32
  %arrayidx9 = getelementptr inbounds i16, i16* %b, i32 2
  %ld.5 = load i16, i16* %arrayidx9, align 2
  %conv10 = sext i16 %ld.5 to i32
  %mul11 = mul nsw i32 %conv10, %conv8
  %add12 = add i32 %mul, %mul11
  %arrayidx13 = getelementptr inbounds i16, i16* %in, i32 -3
  %ld.6 = load i16, i16* %arrayidx13, align 2
  %conv14 = sext i16 %ld.6 to i32
  %arrayidx15 = getelementptr inbounds i16, i16* %b, i32 3
  %ld.7 = load i16, i16* %arrayidx15, align 2
  %conv16 = sext i16 %ld.7 to i32
  %mul17 = mul nsw i32 %conv16, %conv14
  %add18 = add i32 %add12, %mul17
  %arrayidx19 = getelementptr inbounds i16, i16* %in, i32 -4
  %ld.8 = load i16, i16* %arrayidx19, align 2
  %conv20 = sext i16 %ld.8 to i32
  %arrayidx21 = getelementptr inbounds i16, i16* %b, i32 4
  %ld.9 = load i16, i16* %arrayidx21, align 2
  %conv22 = sext i16 %ld.9 to i32
  %mul23 = mul nsw i32 %conv22, %conv20
  %add24 = add i32 %add18, %mul23
  %arrayidx25 = getelementptr inbounds i16, i16* %in, i32 -5
  %ld.10 = load i16, i16* %arrayidx25, align 2
  %conv26 = sext i16 %ld.10 to i32
  %arrayidx27 = getelementptr inbounds i16, i16* %b, i32 5
  %ld.11 = load i16, i16* %arrayidx27, align 2
  %conv28 = sext i16 %ld.11 to i32
  %mul29 = mul nsw i32 %conv28, %conv26
  %add30 = add i32 %add24, %mul29
  ret i32 %add30
}

declare dso_local i32 @bar(i32, i32) local_unnamed_addr

