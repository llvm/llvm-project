; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpTypeInt type with a width other than 8, 16, 32 or 64 bits requires the following SPIR-V extension: SPV_ALTERA_arbitrary_precision_integers

; CHECK: OpCapability ArbitraryPrecisionIntegersALTERA
; CHECK: OpExtension "SPV_ALTERA_arbitrary_precision_integers"
; CHECK: OpName %[[#TestAdd:]] "test_add"
; CHECK: OpName %[[#TestSub:]] "test_sub"
; CHECK: %[[#Int128Ty:]] = OpTypeInt 128 0
; CHECK: %[[#Const64Int128:]] = OpConstant %[[#Int128Ty]] 64 0 0 0

; CHECK: %[[#TestAdd]] = OpFunction
define spir_func void @test_add(i64 %AL, i64 %AH, i64 %BL, i64 %BH, ptr %RL, ptr %RH) {
entry:
; CHECK: {{.*}} = OpUConvert %[[#Int128Ty]]
; CHECK: {{.*}} = OpUConvert %[[#Int128Ty]]
; CHECK: {{.*}} = OpShiftLeftLogical %[[#Int128Ty]] {{%[0-9]+}} %[[#Const64Int128]]
; CHECK: {{.*}} = OpBitwiseOr %[[#Int128Ty]]
; CHECK: {{.*}} = OpUConvert %[[#Int128Ty]]
; CHECK: {{.*}} = OpIAdd %[[#Int128Ty]]
	%tmp1 = zext i64 %AL to i128
	%tmp23 = zext i64 %AH to i128
	%tmp4 = shl i128 %tmp23, 64
	%tmp5 = or i128 %tmp4, %tmp1
	%tmp67 = zext i64 %BL to i128
	%tmp89 = zext i64 %BH to i128
	%tmp11 = shl i128 %tmp89, 64
	%tmp12 = or i128 %tmp11, %tmp67
	%tmp15 = add i128 %tmp12, %tmp5
	%tmp1617 = trunc i128 %tmp15 to i64
	store i64 %tmp1617, ptr %RL
	%tmp21 = lshr i128 %tmp15, 64
	%tmp2122 = trunc i128 %tmp21 to i64
	store i64 %tmp2122, ptr %RH
	ret void
; CHECK: OpFunctionEnd
}

; CHECK: %[[#TestSub]] = OpFunction
define spir_func void @test_sub(i64 %AL, i64 %AH, i64 %BL, i64 %BH, ptr %RL, ptr %RH) {
entry:
; CHECK: {{.*}} = OpUConvert %[[#Int128Ty]]
; CHECK: {{.*}} = OpUConvert %[[#Int128Ty]]
; CHECK: {{.*}} = OpShiftLeftLogical %[[#Int128Ty]] {{%[0-9]+}} %[[#Const64Int128]]
; CHECK: {{.*}} = OpBitwiseOr %[[#Int128Ty]]
; CHECK: {{.*}} = OpUConvert %[[#Int128Ty]]
; CHECK: {{.*}} = OpISub %[[#Int128Ty]]
	%tmp1 = zext i64 %AL to i128
	%tmp23 = zext i64 %AH to i128
	%tmp4 = shl i128 %tmp23, 64
	%tmp5 = or i128 %tmp4, %tmp1
	%tmp67 = zext i64 %BL to i128
	%tmp89 = zext i64 %BH to i128
	%tmp11 = shl i128 %tmp89, 64
	%tmp12 = or i128 %tmp11, %tmp67
	%tmp15 = sub i128 %tmp5, %tmp12
	%tmp1617 = trunc i128 %tmp15 to i64
	store i64 %tmp1617, ptr %RL
	%tmp21 = lshr i128 %tmp15, 64
	%tmp2122 = trunc i128 %tmp21 to i64
	store i64 %tmp2122, ptr %RH
	ret void
; CHECK: OpFunctionEnd
}
