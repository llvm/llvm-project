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
; CHECK: [[ACC:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN_MINUS_3]], i32 [[B_PLUS_2]], i32 [[ADD0]])
; CHECK: [[ADDR_IN_MINUS_5:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -5
; CHECK: [[CAST_ADDR_IN_MINUS_5:%[^ ]+]] = bitcast i16* [[ADDR_IN_MINUS_5]] to i32*
; CHECK: [[IN_MINUS_5:%[^ ]+]] = load i32, i32* [[CAST_ADDR_IN_MINUS_5]], align 2
; CHECK: [[ADDR_B_PLUS_4:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 4
; CHECK: [[CAST_ADDR_B_PLUS_4:%[^ ]+]] = bitcast i16* [[ADDR_B_PLUS_4]] to i32*
; CHECK: [[B_PLUS_4:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_PLUS_4]], align 2
; CHECK: [[RES:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN_MINUS_5]], i32 [[B_PLUS_4]], i32 [[ACC]])
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
; CHECK: [[ACC:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN_MINUS_3]], i32 [[B_PLUS_2]], i32 [[MUL0]])
; CHECK: [[ADDR_IN_MINUS_5:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -5
; CHECK: [[CAST_ADDR_IN_MINUS_5:%[^ ]+]] = bitcast i16* [[ADDR_IN_MINUS_5]] to i32*
; CHECK: [[IN_MINUS_5:%[^ ]+]] = load i32, i32* [[CAST_ADDR_IN_MINUS_5]], align 2
; CHECK: [[ADDR_B_PLUS_4:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 4
; CHECK: [[CAST_ADDR_B_PLUS_4:%[^ ]+]] = bitcast i16* [[ADDR_B_PLUS_4]] to i32*
; CHECK: [[B_PLUS_4:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_PLUS_4]], align 2
; CHECK: [[RES:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[IN_MINUS_5]], i32 [[B_PLUS_4]], i32 [[ACC]])
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

; CHECK-LABEL: with_64bit_acc
; CHECK: [[ADDR_IN_MINUS_1:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -1
; CHECK: [[LD_IN_MINUS_1:%[^ ]+]] = load i16, i16* [[ADDR_IN_MINUS_1]], align 2
; CHECK: [[IN_MINUS_1:%[^ ]+]] = sext i16 [[LD_IN_MINUS_1]] to i32
; CHECK: [[ADDR_B_PLUS_1:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 1
; CHECK: [[LD_B_PLUS_1:%[^ ]+]] = load i16, i16* [[ADDR_B_PLUS_1]], align 2
; CHECK: [[B_PLUS_1:%[^ ]+]] = sext i16 [[LD_B_PLUS_1]] to i32
; CHECK: [[MUL0:%[^ ]+]] = mul nsw i32 [[B_PLUS_1]], [[IN_MINUS_1]]
; CHECK: [[SEXT1:%[^ ]+]] = sext i32 [[MUL0]] to i64
; CHECK: [[ADD0:%[^ ]+]] = add i64 %sext.0, [[SEXT1]]
; CHECK: [[ADDR_IN_MINUS_3:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -3
; CHECK: [[CAST_ADDR_IN_MINUS_3:%[^ ]+]] = bitcast i16* [[ADDR_IN_MINUS_3]] to i32*
; CHECK: [[IN_MINUS_3:%[^ ]+]] = load i32, i32* [[CAST_ADDR_IN_MINUS_3]], align 2
; CHECK: [[ADDR_B_PLUS_2:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 2
; CHECK: [[CAST_ADDR_B_PLUS_2:%[^ ]+]] = bitcast i16* [[ADDR_B_PLUS_2]] to i32*
; CHECK: [[B_PLUS_2:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_PLUS_2]], align 2
; CHECK: [[ACC:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[IN_MINUS_3]], i32 [[B_PLUS_2]], i64 [[ADD0]])
; CHECK: [[ADDR_IN_MINUS_5:%[^ ]+]] = getelementptr inbounds i16, i16* %in, i32 -5
; CHECK: [[CAST_ADDR_IN_MINUS_5:%[^ ]+]] = bitcast i16* [[ADDR_IN_MINUS_5]] to i32*
; CHECK: [[IN_MINUS_5:%[^ ]+]] = load i32, i32* [[CAST_ADDR_IN_MINUS_5]], align 2
; CHECK: [[ADDR_B_PLUS_4:%[^ ]+]] = getelementptr inbounds i16, i16* %b, i32 4
; CHECK: [[CAST_ADDR_B_PLUS_4:%[^ ]+]] = bitcast i16* [[ADDR_B_PLUS_4]] to i32*
; CHECK: [[B_PLUS_4:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_PLUS_4]], align 2
; CHECK: [[RES:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[IN_MINUS_5]], i32 [[B_PLUS_4]], i64 [[ACC]])
; CHECK: ret i64 [[RES]]
define i64 @with_64bit_acc(i16* nocapture readonly %in, i16* nocapture readonly %b) {
entry:
  %0 = load i16, i16* %in, align 2
  %conv = sext i16 %0 to i32
  %1 = load i16, i16* %b, align 2
  %conv2 = sext i16 %1 to i32
  %call = tail call i32 @bar(i32 %conv, i32 %conv2)
  %sext.0 = sext i32 %call to i64
  %arrayidx3 = getelementptr inbounds i16, i16* %in, i32 -1
  %2 = load i16, i16* %arrayidx3, align 2
  %conv4 = sext i16 %2 to i32
  %arrayidx5 = getelementptr inbounds i16, i16* %b, i32 1
  %3 = load i16, i16* %arrayidx5, align 2
  %conv6 = sext i16 %3 to i32
  %mul = mul nsw i32 %conv6, %conv4
  %sext.1 = sext i32 %mul to i64
  %add = add i64 %sext.0, %sext.1
  %arrayidx7 = getelementptr inbounds i16, i16* %in, i32 -2
  %4 = load i16, i16* %arrayidx7, align 2
  %conv8 = sext i16 %4 to i32
  %arrayidx9 = getelementptr inbounds i16, i16* %b, i32 2
  %5 = load i16, i16* %arrayidx9, align 2
  %conv10 = sext i16 %5 to i32
  %mul11 = mul nsw i32 %conv10, %conv8
  %sext.2 = sext i32 %mul11 to i64
  %add12 = add i64 %add, %sext.2
  %arrayidx13 = getelementptr inbounds i16, i16* %in, i32 -3
  %6 = load i16, i16* %arrayidx13, align 2
  %conv14 = sext i16 %6 to i32
  %arrayidx15 = getelementptr inbounds i16, i16* %b, i32 3
  %7 = load i16, i16* %arrayidx15, align 2
  %conv16 = sext i16 %7 to i32
  %mul17 = mul nsw i32 %conv16, %conv14
  %sext.3 = sext i32 %mul17 to i64
  %add18 = add i64 %add12, %sext.3
  %arrayidx19 = getelementptr inbounds i16, i16* %in, i32 -4
  %8 = load i16, i16* %arrayidx19, align 2
  %conv20 = sext i16 %8 to i32
  %arrayidx21 = getelementptr inbounds i16, i16* %b, i32 4
  %9 = load i16, i16* %arrayidx21, align 2
  %conv22 = sext i16 %9 to i32
  %mul23 = mul nsw i32 %conv22, %conv20
  %sext.4 = sext i32 %mul23 to i64
  %add24 = add i64 %add18, %sext.4
  %arrayidx25 = getelementptr inbounds i16, i16* %in, i32 -5
  %10 = load i16, i16* %arrayidx25, align 2
  %conv26 = sext i16 %10 to i32
  %arrayidx27 = getelementptr inbounds i16, i16* %b, i32 5
  %11 = load i16, i16* %arrayidx27, align 2
  %conv28 = sext i16 %11 to i32
  %mul29 = mul nsw i32 %conv28, %conv26
  %sext.5 = sext i32 %mul29 to i64
  %add30 = add i64 %add24, %sext.5
  ret i64 %add30
}

; CHECK: with_64bit_add_acc
; CHECK: [[ADDR_X_PLUS_1:%[^ ]+]] = getelementptr inbounds i16, i16* %px.10756.unr, i32 1
; CHECK: [[X:%[^ ]+]] = load i16, i16* %px.10756.unr, align 2
; CHECK: [[SEXT_X:%[^ ]+]] = sext i16 [[X]] to i32
; CHECK: [[ADDR_Y_MINUS_1:%[^ ]+]] = getelementptr inbounds i16, i16* %py.8757.unr, i32 -1
; CHECK: [[Y:%[^ ]+]] = load i16, i16* %py.8757.unr, align 2
; CHECK: [[SEXT_Y:%[^ ]+]] = sext i16 [[Y]] to i32
; CHECK: [[MUL0:%[^ ]+]] = mul nsw i32 [[SEXT_Y]], [[SEXT_X]]
; CHECK: [[SEXT_MUL0:%[^ ]+]] = sext i32 [[MUL0]] to i64
; CHECK: [[ADD_1:%[^ ]+]] = add nsw i64 %sum.3758.unr, [[SEXT_MUL0]]
; CHECK: [[X_PLUS_2:%[^ ]+]] = getelementptr inbounds i16, i16* %px.10756.unr, i32 2
; CHECK: [[X_1:%[^ ]+]] = load i16, i16* [[ADDR_X_PLUS_1]], align 2
; CHECK: [[SEXT_X_1:%[^ ]+]] = sext i16 [[X_1]] to i32
; CHECK: [[Y_1:%[^ ]+]] = load i16, i16* [[ADDR_Y_MINUS_1]], align 2
; CHECK: [[SEXT_Y_1:%[^ ]+]] = sext i16 [[Y_1]] to i32
; CHECK: [[UNPAIRED:%[^ ]+]] = mul nsw i32 [[SEXT_Y_1]], [[SEXT_X_1]]
; CHECK: [[SEXT:%[^ ]+]] = sext i32 [[UNPAIRED]] to i64
; CHECK: [[ACC:%[^ ]+]] = add i64 [[SEXT]], [[ADD_1]]
; CHECK: [[ADDR_X_PLUS_2:%[^ ]+]] = bitcast i16* [[X_PLUS_2]] to i32*
; CHECK: [[X_2:%[^ ]+]] = load i32, i32* [[ADDR_X_PLUS_2]], align 2
; CHECK: [[Y_MINUS_3:%[^ ]+]] = getelementptr inbounds i16, i16* %py.8757.unr, i32 -3
; CHECK: [[ADDR_Y_MINUS_3:%[^ ]+]] = bitcast i16* [[Y_MINUS_3]] to i32*
; CHECK: [[Y_3:%[^ ]+]] = load i32, i32* [[ADDR_Y_MINUS_3]], align 2
; CHECK: [[RES:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[Y_3]], i32 [[X_2]], i64 [[ACC]])
; CHECK: ret i64 [[RES]]
define i64 @with_64bit_add_acc(i16* nocapture readonly %px.10756.unr, i16* nocapture readonly %py.8757.unr, i32 %acc) {
entry:
  %sum.3758.unr = sext i32 %acc to i64
  br label %bb.1

bb.1:
  %incdec.ptr184.epil = getelementptr inbounds i16, i16* %px.10756.unr, i32 1
  %tmp216 = load i16, i16* %px.10756.unr, align 2
  %conv185.epil = sext i16 %tmp216 to i32
  %incdec.ptr186.epil = getelementptr inbounds i16, i16* %py.8757.unr, i32 -1
  %tmp217 = load i16, i16* %py.8757.unr, align 2
  %conv187.epil = sext i16 %tmp217 to i32
  %mul.epil = mul nsw i32 %conv187.epil, %conv185.epil
  %conv188.epil = sext i32 %mul.epil to i64
  %add189.epil = add nsw i64 %sum.3758.unr, %conv188.epil
  %incdec.ptr190.epil = getelementptr inbounds i16, i16* %px.10756.unr, i32 2
  %tmp218 = load i16, i16* %incdec.ptr184.epil, align 2
  %conv191.epil = sext i16 %tmp218 to i32
  %incdec.ptr192.epil = getelementptr inbounds i16, i16* %py.8757.unr, i32 -2
  %tmp219 = load i16, i16* %incdec.ptr186.epil, align 2
  %conv193.epil = sext i16 %tmp219 to i32
  %mul194.epil = mul nsw i32 %conv193.epil, %conv191.epil
  %conv195.epil = sext i32 %mul194.epil to i64
  %add196.epil = add nsw i64 %add189.epil, %conv195.epil
  %incdec.ptr197.epil = getelementptr inbounds i16, i16* %px.10756.unr, i32 3
  %tmp220 = load i16, i16* %incdec.ptr190.epil, align 2
  %conv198.epil = sext i16 %tmp220 to i32
  %incdec.ptr199.epil = getelementptr inbounds i16, i16* %py.8757.unr, i32 -3
  %tmp221 = load i16, i16* %incdec.ptr192.epil, align 2
  %conv200.epil = sext i16 %tmp221 to i32
  %mul201.epil = mul nsw i32 %conv200.epil, %conv198.epil
  %conv202.epil = sext i32 %mul201.epil to i64
  %add203.epil = add nsw i64 %add196.epil, %conv202.epil
  %tmp222 = load i16, i16* %incdec.ptr197.epil, align 2
  %conv205.epil = sext i16 %tmp222 to i32
  %tmp223 = load i16, i16* %incdec.ptr199.epil, align 2
  %conv207.epil = sext i16 %tmp223 to i32
  %mul208.epil = mul nsw i32 %conv207.epil, %conv205.epil
  %conv209.epil = sext i32 %mul208.epil to i64
  %add210.epil = add nsw i64 %add203.epil, %conv209.epil
  ret i64 %add210.epil
}

declare dso_local i32 @bar(i32, i32) local_unnamed_addr

