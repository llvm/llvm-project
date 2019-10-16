; RUN: opt -arm-parallel-dsp -mtriple=armv7-a -S %s -o - | FileCheck %s

; CHECK-LABEL: overlap_1
; CHECK: [[ADDR_A_1:%[^ ]+]] = getelementptr i16, i16* %a, i32 1
; CHECK: [[ADDR_B_1:%[^ ]+]] = getelementptr i16, i16* %b, i32 1
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[ACC:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[LD_A]], i32 [[LD_B]], i32 %acc)
; CHECK: [[CAST_A_1:%[^ ]+]] = bitcast i16* [[ADDR_A_1]] to i32*
; CHECK: [[LD_A_1:%[^ ]+]] = load i32, i32* [[CAST_A_1]]
; CHECK: [[CAST_B_1:%[^ ]+]] = bitcast i16* [[ADDR_B_1]] to i32*
; CHECK: [[LD_B_1:%[^ ]+]] = load i32, i32* [[CAST_B_1]]
; CHECK: [[RES:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[LD_A_1]], i32 [[LD_B_1]], i32 [[ACC]])
; CHECK: ret i32 [[RES]]
define i32 @overlap_1(i16* %a, i16* %b, i32 %acc) {
entry:
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %ld.a.1 = load i16, i16* %addr.a.1
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.b.2 = load i16, i16* %addr.b.2
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %mul.2 = mul i32 %sext.a.2, %sext.b.2
  %add = add i32 %mul.0, %mul.1
  %add.1 = add i32 %mul.1, %mul.2
  %add.2 = add i32 %add.1, %add
  %res = add i32 %add.2, %acc
  ret i32 %res
}

; TODO: Is it really best to generate smlald for the first instruction? Does
; this just increase register pressure unnecessarily?
; CHECK-LABEL: overlap_64_1
; CHECK: [[ADDR_A_1:%[^ ]+]] = getelementptr i16, i16* %a, i32 1
; CHECK: [[ADDR_B_1:%[^ ]+]] = getelementptr i16, i16* %b, i32 1
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[ACC:%[^ ]+]] = call i64 @llvm.arm.smlald(i32 [[LD_A]], i32 [[LD_B]], i64 %acc)
; CHECK: [[CAST_A_1:%[^ ]+]] = bitcast i16* [[ADDR_A_1]] to i32*
; CHECK: [[LD_A_1:%[^ ]+]] = load i32, i32* [[CAST_A_1]]
; CHECK: [[CAST_B_1:%[^ ]+]] = bitcast i16* [[ADDR_B_1]] to i32*
; CHECK: [[LD_B_1:%[^ ]+]] = load i32, i32* [[CAST_B_1]]
; CHECK: [[RES:%[^ ]+]] = call i64 @llvm.arm.smlald(i32 [[LD_A_1]], i32 [[LD_B_1]], i64 [[ACC]])
; CHECK: ret i64 [[RES]]
define i64 @overlap_64_1(i16* %a, i16* %b, i64 %acc) {
entry:
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %ld.a.1 = load i16, i16* %addr.a.1
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.b.2 = load i16, i16* %addr.b.2
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %mul.2 = mul i32 %sext.a.2, %sext.b.2
  %add = add i32 %mul.0, %mul.1
  %add.1 = add i32 %mul.1, %mul.2
  %sext.add = sext i32 %add to i64
  %sext.add.1 = sext i32 %add.1 to i64
  %add.2 = add i64 %sext.add.1, %sext.add
  %res = add i64 %add.2, %acc
  ret i64 %res
}

; CHECK-LABEL: overlap_2
; CHECK: [[ADDR_A_1:%[^ ]+]] = getelementptr i16, i16* %a, i32 1
; CHECK: [[ADDR_B_1:%[^ ]+]] = getelementptr i16, i16* %b, i32 1
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[ACC1:%[^ ]+]] = add i32 %mul.1, %acc
; CHECK: [[ACC2:%[^ ]+]] = add i32 %mul.2, [[ACC1]]
; CHECK: [[RES:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[LD_A]], i32 [[LD_B]], i32 [[ACC2]])
; CHECK: ret i32 [[RES]]
define i32 @overlap_2(i16* %a, i16* %b, i32 %acc) {
entry:
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %ld.a.1 = load i16, i16* %addr.a.1
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.b.2 = load i16, i16* %addr.b.2
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %mul.2 = mul i32 %sext.b.2, %sext.a.2
  %add = add i32 %mul.0, %mul.1
  %add.1 = add i32 %mul.1, %mul.2
  %add.2 = add i32 %add, %add.1
  %res = add i32 %add.2, %acc
  ret i32 %res
}

; CHECK-LABEL: overlap_3
; CHECK: [[GEP_B:%[^ ]+]] = getelementptr i16, i16* %b, i32 1
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[SMLAD:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[LD_A]], i32 [[LD_B]], i32 %acc)
; CHECK: [[CAST_B_1:%[^ ]+]] = bitcast i16* [[GEP_B]] to i32*
; CHECK: [[LD_B_1:%[^ ]+]] = load i32, i32* [[CAST_B_1]]
; CHECK: [[GEP_A:%[^ ]+]] = getelementptr i16, i16* %a, i32 2
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* [[GEP_A]] to i32*
; CHECK: [[LD_A_2:%[^ ]+]] = load i32, i32* [[CAST_A_2]]
; CHECK: [[RES:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[LD_A_2]], i32 [[LD_B_1]], i32 [[SMLAD]])
; CHECK: ret i32 [[RES]]
define i32 @overlap_3(i16* %a, i16* %b, i32 %acc) {
entry:
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %ld.a.1 = load i16, i16* %addr.a.1
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.b.2 = load i16, i16* %addr.b.2
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %mul.2 = mul i32 %sext.a.2, %sext.b.1
  %mul.3 = mul i32 %sext.a.3, %sext.b.2
  %add = add i32 %mul.0, %mul.1
  %add.1 = add i32 %mul.2, %mul.3
  %add.2 = add i32 %add.1, %add
  %res = add i32 %add.2, %acc
  ret i32 %res
}

; CHECK-LABEL: overlap_4
; CHECK: [[GEP_B:%[^ ]+]] = getelementptr i16, i16* %b, i32 1
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[SMLAD:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[LD_A]], i32 [[LD_B]], i32 %acc)
; CHECK: [[CAST_B_1:%[^ ]+]] = bitcast i16* [[GEP_B]] to i32*
; CHECK: [[LD_B_1:%[^ ]+]] = load i32, i32* [[CAST_B_1]]
; CHECK: [[GEP_A:%[^ ]+]] = getelementptr i16, i16* %a, i32 2
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* [[GEP_A]] to i32*
; CHECK: [[LD_A_2:%[^ ]+]] = load i32, i32* [[CAST_A_2]]
; CHECK: [[RES:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[LD_A_2]], i32 [[LD_B_1]], i32 [[SMLAD]])
; CHECK: ret i32 [[RES]]
define i32 @overlap_4(i16* %a, i16* %b, i32 %acc) {
entry:
  %addr.a.1 = getelementptr i16, i16* %a, i32 1
  %addr.b.1 = getelementptr i16, i16* %b, i32 1
  %ld.a.0 = load i16, i16* %a
  %sext.a.0 = sext i16 %ld.a.0 to i32
  %ld.b.0 = load i16, i16* %b
  %ld.a.1 = load i16, i16* %addr.a.1
  %ld.b.1 = load i16, i16* %addr.b.1
  %sext.a.1 = sext i16 %ld.a.1 to i32
  %sext.b.1 = sext i16 %ld.b.1 to i32
  %sext.b.0 = sext i16 %ld.b.0 to i32
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.b.2 = getelementptr i16, i16* %b, i32 2
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.b.2 = load i16, i16* %addr.b.2
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.b.2 = sext i16 %ld.b.2 to i32
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %mul.2 = mul i32 %sext.b.2, %sext.a.2
  %mul.3 = mul i32 %sext.b.1, %sext.a.3
  %add = add i32 %mul.0, %mul.1
  %add.1 = add i32 %mul.2, %mul.3
  %add.2 = add i32 %add.1, %add
  %res = add i32 %add.2, %acc
  ret i32 %res
}
