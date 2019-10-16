; RUN: llc -mtriple=thumbv7em -mcpu=cortex-m4 -O3 %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LLC
; RUN: opt -S -mtriple=armv7-a -arm-parallel-dsp -dce %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-OPT

; TODO: Think we should be able to use smlsdx/smlsldx here.

; CHECK-LABEL: complex_dot_prod

; CHECK-LLC: smlaldx
; CHECK-LLC: smulbb
; CHECK-LLC: smultt
; CHECK-LLC: smlaldx
; CHECK-LLC: smlalbb
; CHECK-LLC: smultt
; CHECK-LLC: smlalbb
; CHECK-LLC: smultt
; CHECK-LLC: smlaldx
; CHECK-LLC: smlalbb
; CHECK-LLC: smultt
; CHECK-LLC: smlaldx
; CHECK-LCC: pop.w {r4, r5, r6, r7, r8, r9, r10, pc}

; CHECK-OPT: [[ADDR_A:%[^ ]+]] = bitcast i16* %pSrcA to i32*
; CHECK-OPT: [[A:%[^ ]+]] = load i32, i32* [[ADDR_A]], align 2
; CHECK-OPT: [[ADDR_A_2:%[^ ]+]] = getelementptr inbounds i16, i16* %pSrcA, i32 2
; CHECK-OPT: [[ADDR_B:%[^ ]+]] = bitcast i16* %pSrcB to i32*
; CHECK-OPT: [[B:%[^ ]+]] = load i32, i32* [[ADDR_B]], align 2
; CHECK-OPT: [[ACC0:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[A]], i32 [[B]], i64 0)
; CHECK-OPT: [[ADDR_B_2:%[^ ]+]] = getelementptr inbounds i16, i16* %pSrcB, i32 2
; CHECK-OPT: [[CAST_ADDR_A_2:%[^ ]+]] = bitcast i16* [[ADDR_A_2]] to i32*
; CHECK-OPT: [[A_2:%[^ ]+]] = load i32, i32* [[CAST_ADDR_A_2]], align 2
; CHECK-OPT: [[ADDR_A_4:%[^ ]+]] = getelementptr inbounds i16, i16* %pSrcA, i32 4
; CHECK-OPT: [[CAST_ADDR_B_2:%[^ ]+]] = bitcast i16* [[ADDR_B_2]] to i32*
; CHECK-OPT: [[B_2:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_2]], align 2
; CHECK-OPT: [[ACC1:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[A_2]], i32 [[B_2]], i64 [[ACC0]])
; CHECK-OPT: [[ADDR_B_4:%[^ ]+]] = getelementptr inbounds i16, i16* %pSrcB, i32 4
; CHECK-OPT: [[CAST_ADDR_A_4:%[^ ]+]] = bitcast i16* [[ADDR_A_4]] to i32*
; CHECK-OPT: [[A_4:%[^ ]+]] = load i32, i32* [[CAST_ADDR_A_4]], align 2
; CHECK-OPT: [[ADDR_A_6:%[^ ]+]] = getelementptr inbounds i16, i16* %pSrcA, i32 6
; CHECK-OPT: [[CAST_ADDR_B_4:%[^ ]+]] = bitcast i16* [[ADDR_B_4]] to i32*
; CHECK-OPT: [[B_4:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_4]], align 2
; CHECK-OPT: [[ACC2:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[A_4]], i32 [[B_4]], i64 [[ACC1]])
; CHECK-OPT: [[ADDR_B_6:%[^ ]+]] = getelementptr inbounds i16, i16* %pSrcB, i32 6
; CHECK-OPT: [[CAST_ADDR_A_6:%[^ ]+]] = bitcast i16* [[ADDR_A_6]] to i32*
; CHECK-OPT: [[A_6:%[^ ]+]] = load i32, i32* [[CAST_ADDR_A_6]], align 2
; CHECK-OPT: [[CAST_ADDR_B_6:%[^ ]+]] = bitcast i16* [[ADDR_B_6]] to i32*
; CHECK-OPT: [[B_6:%[^ ]+]] = load i32, i32* [[CAST_ADDR_B_6]], align 2
; CHECK-OPT: call i64 @llvm.arm.smlaldx(i32 [[A_6]], i32 [[B_6]], i64 [[ACC2]])

define dso_local arm_aapcscc void @complex_dot_prod(i16* nocapture readonly %pSrcA, i16* nocapture readonly %pSrcB, i32* nocapture %realResult, i32* nocapture %imagResult) {
entry:
  %incdec.ptr = getelementptr inbounds i16, i16* %pSrcA, i32 1
  %0 = load i16, i16* %pSrcA, align 2
  %incdec.ptr1 = getelementptr inbounds i16, i16* %pSrcA, i32 2
  %1 = load i16, i16* %incdec.ptr, align 2
  %incdec.ptr2 = getelementptr inbounds i16, i16* %pSrcB, i32 1
  %2 = load i16, i16* %pSrcB, align 2
  %incdec.ptr3 = getelementptr inbounds i16, i16* %pSrcB, i32 2
  %3 = load i16, i16* %incdec.ptr2, align 2
  %conv = sext i16 %0 to i32
  %conv4 = sext i16 %2 to i32
  %mul = mul nsw i32 %conv4, %conv
  %conv5 = sext i32 %mul to i64
  %conv7 = sext i16 %3 to i32
  %mul8 = mul nsw i32 %conv7, %conv
  %conv9 = sext i32 %mul8 to i64
  %conv11 = sext i16 %1 to i32
  %mul13 = mul nsw i32 %conv7, %conv11
  %conv14 = sext i32 %mul13 to i64
  %sub = sub nsw i64 %conv5, %conv14
  %mul17 = mul nsw i32 %conv4, %conv11
  %conv18 = sext i32 %mul17 to i64
  %add19 = add nsw i64 %conv9, %conv18
  %incdec.ptr20 = getelementptr inbounds i16, i16* %pSrcA, i32 3
  %4 = load i16, i16* %incdec.ptr1, align 2
  %incdec.ptr21 = getelementptr inbounds i16, i16* %pSrcA, i32 4
  %5 = load i16, i16* %incdec.ptr20, align 2
  %incdec.ptr22 = getelementptr inbounds i16, i16* %pSrcB, i32 3
  %6 = load i16, i16* %incdec.ptr3, align 2
  %incdec.ptr23 = getelementptr inbounds i16, i16* %pSrcB, i32 4
  %7 = load i16, i16* %incdec.ptr22, align 2
  %conv24 = sext i16 %4 to i32
  %conv25 = sext i16 %6 to i32
  %mul26 = mul nsw i32 %conv25, %conv24
  %conv27 = sext i32 %mul26 to i64
  %add28 = add nsw i64 %sub, %conv27
  %conv30 = sext i16 %7 to i32
  %mul31 = mul nsw i32 %conv30, %conv24
  %conv32 = sext i32 %mul31 to i64
  %conv34 = sext i16 %5 to i32
  %mul36 = mul nsw i32 %conv30, %conv34
  %conv37 = sext i32 %mul36 to i64
  %sub38 = sub nsw i64 %add28, %conv37
  %mul41 = mul nsw i32 %conv25, %conv34
  %conv42 = sext i32 %mul41 to i64
  %add33 = add nsw i64 %add19, %conv42
  %add43 = add nsw i64 %add33, %conv32
  %incdec.ptr44 = getelementptr inbounds i16, i16* %pSrcA, i32 5
  %8 = load i16, i16* %incdec.ptr21, align 2
  %incdec.ptr45 = getelementptr inbounds i16, i16* %pSrcA, i32 6
  %9 = load i16, i16* %incdec.ptr44, align 2
  %incdec.ptr46 = getelementptr inbounds i16, i16* %pSrcB, i32 5
  %10 = load i16, i16* %incdec.ptr23, align 2
  %incdec.ptr47 = getelementptr inbounds i16, i16* %pSrcB, i32 6
  %11 = load i16, i16* %incdec.ptr46, align 2
  %conv48 = sext i16 %8 to i32
  %conv49 = sext i16 %10 to i32
  %mul50 = mul nsw i32 %conv49, %conv48
  %conv51 = sext i32 %mul50 to i64
  %add52 = add nsw i64 %sub38, %conv51
  %conv54 = sext i16 %11 to i32
  %mul55 = mul nsw i32 %conv54, %conv48
  %conv56 = sext i32 %mul55 to i64
  %conv58 = sext i16 %9 to i32
  %mul60 = mul nsw i32 %conv54, %conv58
  %conv61 = sext i32 %mul60 to i64
  %sub62 = sub nsw i64 %add52, %conv61
  %mul65 = mul nsw i32 %conv49, %conv58
  %conv66 = sext i32 %mul65 to i64
  %add57 = add nsw i64 %add43, %conv66
  %add67 = add nsw i64 %add57, %conv56
  %incdec.ptr68 = getelementptr inbounds i16, i16* %pSrcA, i32 7
  %12 = load i16, i16* %incdec.ptr45, align 2
  %13 = load i16, i16* %incdec.ptr68, align 2
  %incdec.ptr70 = getelementptr inbounds i16, i16* %pSrcB, i32 7
  %14 = load i16, i16* %incdec.ptr47, align 2
  %15 = load i16, i16* %incdec.ptr70, align 2
  %conv72 = sext i16 %12 to i32
  %conv73 = sext i16 %14 to i32
  %mul74 = mul nsw i32 %conv73, %conv72
  %conv75 = sext i32 %mul74 to i64
  %add76 = add nsw i64 %sub62, %conv75
  %conv78 = sext i16 %15 to i32
  %mul79 = mul nsw i32 %conv78, %conv72
  %conv80 = sext i32 %mul79 to i64
  %conv82 = sext i16 %13 to i32
  %mul84 = mul nsw i32 %conv78, %conv82
  %conv85 = sext i32 %mul84 to i64
  %sub86 = sub nsw i64 %add76, %conv85
  %mul89 = mul nsw i32 %conv73, %conv82
  %conv90 = sext i32 %mul89 to i64
  %add81 = add nsw i64 %add67, %conv90
  %add91 = add nsw i64 %add81, %conv80
  %16 = lshr i64 %sub86, 6
  %conv92 = trunc i64 %16 to i32
  store i32 %conv92, i32* %realResult, align 4
  %17 = lshr i64 %add91, 6
  %conv94 = trunc i64 %17 to i32
  store i32 %conv94, i32* %imagResult, align 4
  ret void
}
