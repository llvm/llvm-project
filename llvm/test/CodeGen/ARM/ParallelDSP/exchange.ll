; RUN: opt -arm-parallel-dsp -mtriple=armv7-a -S %s -o - | FileCheck %s

; CHECK-LABEL: exchange_1
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: call i32 @llvm.arm.smladx(i32 [[LD_A]], i32 [[LD_B]]
define i32 @exchange_1(i16* %a, i16* %b, i32 %acc) {
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
  %mul.0 = mul i32 %sext.a.0, %sext.b.1
  %mul.1 = mul i32 %sext.a.1, %sext.b.0
  %add = add i32 %mul.0, %mul.1
  %res = add i32 %add, %acc
  ret i32 %res
}

; CHECK-LABEL: exchange_2
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: call i32 @llvm.arm.smladx(i32 [[LD_A]], i32 [[LD_B]]
define i32 @exchange_2(i16* %a, i16* %b, i32 %acc) {
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
  %mul.0 = mul i32 %sext.b.1, %sext.a.0
  %mul.1 = mul i32 %sext.b.0, %sext.a.1
  %add = add i32 %mul.0, %mul.1
  %res = add i32 %add, %acc
  ret i32 %res
}

; CHECK-LABEL: exchange_3
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: call i32 @llvm.arm.smladx(i32 [[LD_B]], i32 [[LD_A]]
define i32 @exchange_3(i16* %a, i16* %b, i32 %acc) {
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
  %mul.0 = mul i32 %sext.a.0, %sext.b.1
  %mul.1 = mul i32 %sext.a.1, %sext.b.0
  %add = add i32 %mul.1, %mul.0
  %res = add i32 %add, %acc
  ret i32 %res
}

; CHECK-LABEL: exchange_4
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: call i32 @llvm.arm.smladx(i32 [[LD_B]], i32 [[LD_A]]
define i32 @exchange_4(i16* %a, i16* %b, i32 %acc) {
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
  %mul.0 = mul i32 %sext.b.1, %sext.a.0
  %mul.1 = mul i32 %sext.b.0, %sext.a.1
  %add = add i32 %mul.1, %mul.0
  %res = add i32 %add, %acc
  ret i32 %res
}

; CHECK-LABEL: exchange_multi_use_1
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[X:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[LD_A]], i32 [[LD_B]], i32 %acc
; CHECK: [[GEP:%[^ ]+]] = getelementptr i16, i16* %a, i32 2
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* [[GEP]] to i32*
; CHECK: [[LD_A_2:%[^ ]+]] = load i32, i32* [[CAST_A_2]]
; CHECK: call i32 @llvm.arm.smlad(i32 [[LD_A_2]], i32 [[LD_B]], i32 [[X]])
define i32 @exchange_multi_use_1(i16* %a, i16* %b, i32 %acc) {
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
  %mul.0 = mul i32 %sext.a.0, %sext.b.1
  %mul.1 = mul i32 %sext.a.1, %sext.b.0
  %add = add i32 %mul.0, %mul.1
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %mul.2 = mul i32 %sext.a.3, %sext.b.1
  %mul.3 = mul i32 %sext.a.2, %sext.b.0
  %add.1 = add i32 %mul.2, %mul.3
  %add.2 = add i32 %add, %add.1
  %res = add i32 %add.2, %acc
  ret i32 %res
}

; CHECK-LABEL: exchange_multi_use_64_1
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[X:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[LD_A]], i32 [[LD_B]], i64 %acc
; CHECK: [[GEP:%[^ ]+]] = getelementptr i16, i16* %a, i32 2
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* [[GEP]] to i32*
; CHECK: [[LD_A_2:%[^ ]+]] = load i32, i32* [[CAST_A_2]]
; CHECK: call i64 @llvm.arm.smlald(i32 [[LD_A_2]], i32 [[LD_B]], i64 [[X]])
define i64 @exchange_multi_use_64_1(i16* %a, i16* %b, i64 %acc) {
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
  %mul.0 = mul i32 %sext.a.0, %sext.b.1
  %mul.1 = mul i32 %sext.a.1, %sext.b.0
  %add = add i32 %mul.0, %mul.1
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %mul.2 = mul i32 %sext.a.3, %sext.b.1
  %mul.3 = mul i32 %sext.a.2, %sext.b.0
  %add.1 = add i32 %mul.2, %mul.3
  %add.2 = add i32 %add, %add.1
  %sext.add.2 = sext i32 %add.2 to i64
  %res = add i64 %sext.add.2, %acc
  ret i64 %res
}

; CHECK-LABEL: exchange_multi_use_64_2
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[X:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[LD_A]], i32 [[LD_B]], i64 %acc
; CHECK: [[GEP:%[^ ]+]] = getelementptr i16, i16* %a, i32 2
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* [[GEP]] to i32*
; CHECK: [[LD_A_2:%[^ ]+]] = load i32, i32* [[CAST_A_2]]
; CHECK: call i64 @llvm.arm.smlald(i32 [[LD_A_2]], i32 [[LD_B]], i64 [[X]])
define i64 @exchange_multi_use_64_2(i16* %a, i16* %b, i64 %acc) {
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
  %mul.0 = mul i32 %sext.a.0, %sext.b.1
  %mul.1 = mul i32 %sext.a.1, %sext.b.0
  %add = add i32 %mul.0, %mul.1
  %sext.add = sext i32 %add to i64
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %mul.2 = mul i32 %sext.a.3, %sext.b.1
  %mul.3 = mul i32 %sext.a.2, %sext.b.0
  %add.1 = add i32 %mul.2, %mul.3
  %sext.add.1 = sext i32 %add.1 to i64
  %add.2 = add i64 %sext.add, %sext.add.1
  %res = add i64 %add.2, %acc
  ret i64 %res
}

; CHECK-LABEL: exchange_multi_use_2
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[X:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[LD_A]], i32 [[LD_B]], i32 %acc
; CHECK: [[GEP:%[^ ]+]] = getelementptr i16, i16* %a, i32 2
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* [[GEP]] to i32*
; CHECK: [[LD_A_2:%[^ ]+]] = load i32, i32* [[CAST_A_2]]
; CHECK: call i32 @llvm.arm.smladx(i32 [[LD_B]], i32 [[LD_A_2]], i32 [[X]])
define i32 @exchange_multi_use_2(i16* %a, i16* %b, i32 %acc) {
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
  %add = add i32 %mul.0, %mul.1
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %mul.2 = mul i32 %sext.b.0, %sext.a.3
  %mul.3 = mul i32 %sext.b.1, %sext.a.2
  %add.1 = add i32 %mul.2, %mul.3
  %add.2 = add i32 %add, %add.1
  %res = add i32 %add.2, %acc
  ret i32 %res
}

; TODO: Why aren't two intrinsics generated?
; CHECK-LABEL: exchange_multi_use_3
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[GEP:%[^ ]+]] = getelementptr i16, i16* %a, i32 2
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* [[GEP]] to i32*
; CHECK: [[LD_A_2:%[^ ]+]] = load i32, i32* [[CAST_A_2]]
; CHECK-NOT: call i32 @llvm.arm.smlad
; CHECK: [[X:%[^ ]+]] = call i32 @llvm.arm.smladx(i32 [[LD_B]], i32 [[LD_A_2]], i32 0
define i32 @exchange_multi_use_3(i16* %a, i16* %b, i32 %acc) {
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
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %mul.2 = mul i32 %sext.b.0, %sext.a.3
  %mul.3 = mul i32 %sext.b.1, %sext.a.2
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %add = add i32 %mul.0, %mul.1
  %add.1 = add i32 %mul.2, %mul.3
  %sub = sub i32 %add, %add.1
  %res = add i32 %acc, %sub
  ret i32 %res
}

; TODO: Would it be better to generate a smlad and then sign extend it?
; CHECK-LABEL: exchange_multi_use_64_3
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[GEP:%[^ ]+]] = getelementptr i16, i16* %a, i32 2
; CHECK: [[CAST_A_2:%[^ ]+]] = bitcast i16* [[GEP]] to i32*
; CHECK: [[LD_A_2:%[^ ]+]] = load i32, i32* [[CAST_A_2]]
; CHECK: [[ACC:%[^ ]+]] = call i64 @llvm.arm.smlaldx(i32 [[LD_B]], i32 [[LD_A_2]], i64 0)
; CHECK: [[X:%[^ ]+]] = call i64 @llvm.arm.smlald(i32 [[LD_A]], i32 [[LD_B]], i64 [[ACC]])
define i64 @exchange_multi_use_64_3(i16* %a, i16* %b, i64 %acc) {
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
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %mul.2 = mul i32 %sext.b.0, %sext.a.3
  %mul.3 = mul i32 %sext.b.1, %sext.a.2
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %add = add i32 %mul.0, %mul.1
  %add.1 = add i32 %mul.2, %mul.3
  %sext.add = sext i32 %add to i64
  %sext.add.1 = sext i32 %add.1 to i64
  %add.2 = add i64 %sext.add, %sext.add.1
  %res = sub i64 %acc, %add.2
  ret i64 %res
}

; TODO: Why isn't smladx generated too?
; CHECK-LABEL: exchange_multi_use_4
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: [[X:%[^ ]+]] = call i32 @llvm.arm.smlad(i32 [[LD_A]], i32 [[LD_B]], i32 0
; CHECK-NOT: call i32 @llvm.arm.smlad
define i32 @exchange_multi_use_4(i16* %a, i16* %b, i32 %acc) {
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
  %addr.a.2 = getelementptr i16, i16* %a, i32 2
  %addr.a.3 = getelementptr i16, i16* %a, i32 3
  %ld.a.2 = load i16, i16* %addr.a.2
  %ld.a.3 = load i16, i16* %addr.a.3
  %sext.a.2 = sext i16 %ld.a.2 to i32
  %sext.a.3 = sext i16 %ld.a.3 to i32
  %mul.2 = mul i32 %sext.b.0, %sext.a.3
  %mul.3 = mul i32 %sext.b.1, %sext.a.2
  %mul.0 = mul i32 %sext.a.0, %sext.b.0
  %mul.1 = mul i32 %sext.a.1, %sext.b.1
  %add.1 = add i32 %mul.2, %mul.3
  %add = add i32 %mul.0, %mul.1
  %sub = sub i32 %add, %add.1
  %res = add i32 %acc, %sub
  ret i32 %res
}

; CHECK-LABEL: exchange_swap
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: call i32 @llvm.arm.smladx(i32 [[LD_B]], i32 [[LD_A]]
define i32 @exchange_swap(i16* %a, i16* %b, i32 %acc) {
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
  %mul.0 = mul i32 %sext.a.1, %sext.b.0
  %mul.1 = mul i32 %sext.a.0, %sext.b.1
  %add = add i32 %mul.0, %mul.1
  %res = add i32 %add, %acc
  ret i32 %res
}

; CHECK-LABEL: exchange_swap_2
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: call i32 @llvm.arm.smladx(i32 [[LD_A]], i32 [[LD_B]]
define i32 @exchange_swap_2(i16* %a, i16* %b, i32 %acc) {
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
  %mul.0 = mul i32 %sext.a.1, %sext.b.0
  %mul.1 = mul i32 %sext.a.0, %sext.b.1
  %add = add i32 %mul.1, %mul.0
  %res = add i32 %add, %acc
  ret i32 %res
}

; CHECK-LABEL: exchange_swap_3
; CHECK: [[CAST_A:%[^ ]+]] = bitcast i16* %a to i32*
; CHECK: [[LD_A:%[^ ]+]] = load i32, i32* [[CAST_A]]
; CHECK: [[CAST_B:%[^ ]+]] = bitcast i16* %b to i32*
; CHECK: [[LD_B:%[^ ]+]] = load i32, i32* [[CAST_B]]
; CHECK: call i32 @llvm.arm.smladx(i32 [[LD_A]], i32 [[LD_B]]
define i32 @exchange_swap_3(i16* %a, i16* %b, i32 %acc) {
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
  %mul.0 = mul i32 %sext.b.0, %sext.a.1
  %mul.1 = mul i32 %sext.b.1, %sext.a.0
  %add = add i32 %mul.1, %mul.0
  %res = add i32 %add, %acc
  ret i32 %res
}
