; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare <2 x double> @llvm.s390.vfidb(<2 x double>, i32, i32)
define void @test_vfidb(<2 x double> %arg0, i32 %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: %ret0 = call <2 x double> @llvm.s390.vfidb(<2 x double> %arg0, i32 %arg1, i32 0)
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret1 = call <2 x double> @llvm.s390.vfidb(<2 x double> %arg0, i32 0, i32 %arg2)
  %ret0 = call <2 x double> @llvm.s390.vfidb(<2 x double> %arg0, i32 %arg1, i32 0)
  %ret1 = call <2 x double> @llvm.s390.vfidb(<2 x double> %arg0, i32 0, i32 %arg2)
  ret void
}

declare <2 x double> @llvm.s390.vfmaxdb(<2 x double>, <2 x double>, i32)
define <2 x double> @test_vfmaxdb(<2 x double> %arg0, <2 x double> %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret = call <2 x double> @llvm.s390.vfmaxdb(<2 x double> %arg0, <2 x double> %arg1, i32 %arg2)
  %ret = call <2 x double> @llvm.s390.vfmaxdb(<2 x double> %arg0, <2 x double> %arg1, i32 %arg2)
  ret <2 x double> %ret
}

declare <2 x double> @llvm.s390.vfmindb(<2 x double>, <2 x double>, i32)
define <2 x double> @test_vfmindb(<2 x double> %arg0, <2 x double> %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret = call <2 x double> @llvm.s390.vfmindb(<2 x double> %arg0, <2 x double> %arg1, i32 %arg2)
  %ret = call <2 x double> @llvm.s390.vfmindb(<2 x double> %arg0, <2 x double> %arg1, i32 %arg2)
  ret <2 x double> %ret
}

declare <2 x float> @llvm.s390.vfmaxsb(<2 x float>, <2 x float>, i32)
define <2 x float> @test_vfmaxsb(<2 x float> %arg0, <2 x float> %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret = call <2 x float> @llvm.s390.vfmaxsb(<2 x float> %arg0, <2 x float> %arg1, i32 %arg2)
  %ret = call <2 x float> @llvm.s390.vfmaxsb(<2 x float> %arg0, <2 x float> %arg1, i32 %arg2)
  ret <2 x float> %ret
}

declare <2 x float> @llvm.s390.vfminsb(<2 x float>, <2 x float>, i32)
define <2 x float> @test_vfminsb(<2 x float> %arg0, <2 x float> %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret = call <2 x float> @llvm.s390.vfminsb(<2 x float> %arg0, <2 x float> %arg1, i32 %arg2)
  %ret = call <2 x float> @llvm.s390.vfminsb(<2 x float> %arg0, <2 x float> %arg1, i32 %arg2)
  ret <2 x float> %ret
}

declare <4 x float> @llvm.s390.vfisb(<4 x float>, i32, i32)
define <4 x float> @test_vfisb(<4 x float> %arg0, i32 %arg1, i32 %arg2) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg1
  ; CHECK-NEXT: %ret0 = call <4 x float> @llvm.s390.vfisb(<4 x float> %arg0, i32 %arg1, i32 0)
  %ret0 = call <4 x float> @llvm.s390.vfisb(<4 x float> %arg0, i32 %arg1, i32 0)

  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg2
  ; CHECK-NEXT: %ret1 = call <4 x float> @llvm.s390.vfisb(<4 x float> %ret0, i32 0, i32 %arg2)
  %ret1 = call <4 x float> @llvm.s390.vfisb(<4 x float> %ret0, i32 0, i32 %arg2)

  ret <4 x float> %ret1
}

declare <16 x i8> @llvm.s390.vstrcb(<16 x i8>, <16 x i8>, <16 x i8>, i32)
define <16 x i8> @test_vstrcb(<16 x i8> %arg0, <16 x i8> %arg1, <16 x i8> %arg2, i32 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg3
  ; CHECK-NEXT: %ret = call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %arg0, <16 x i8> %arg1, <16 x i8> %arg2, i32 %arg3)
  %ret = call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %arg0, <16 x i8>%arg1, <16 x i8> %arg2, i32 %arg3)
  ret <16 x i8> %ret
}

declare <8 x i16> @llvm.s390.vstrch(<8 x i16>, <8 x i16>, <8 x i16>, i32)
define <8 x i16> @test_vstrch(<8 x i16> %arg0, <8 x i16> %arg1, <8 x i16> %arg2, i32 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg3
  ; CHECK-NEXT: %ret = call <8 x i16> @llvm.s390.vstrch(<8 x i16> %arg0, <8 x i16> %arg1, <8 x i16> %arg2, i32 %arg3)
  %ret = call <8 x i16> @llvm.s390.vstrch(<8 x i16> %arg0, <8 x i16>%arg1, <8 x i16> %arg2, i32 %arg3)
  ret <8 x i16> %ret
}

declare <4 x i32> @llvm.s390.vstrcf(<4 x i32>, <4 x i32>, <4 x i32>, i32)
define <4 x i32> @test_vstrcf(<4 x i32> %arg0, <4 x i32> %arg1, <4 x i32> %arg2, i32 %arg3) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %arg3
  ; CHECK-NEXT: %ret = call <4 x i32> @llvm.s390.vstrcf(<4 x i32> %arg0, <4 x i32> %arg1, <4 x i32> %arg2, i32 %arg3)
  %ret = call <4 x i32> @llvm.s390.vstrcf(<4 x i32> %arg0, <4 x i32>%arg1, <4 x i32> %arg2, i32 %arg3)
  ret <4 x i32> %ret
}

declare <16 x i8> @llvm.s390.verimb(<16 x i8>, <16 x i8>, <16 x i8>, i32)
define <16 x i8> @test_verimb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, i32 %d) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %d
  ; CHECK-NEXT: %res = call <16 x i8> @llvm.s390.verimb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, i32 %d)
  %res = call <16 x i8> @llvm.s390.verimb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, i32 %d)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.s390.verimh(<8 x i16>, <8 x i16>, <8 x i16>, i32)
define <8 x i16> @test_verimh(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c, i32 %d) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %d
  ; CHECK-NEXT: %res = call <8 x i16> @llvm.s390.verimh(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c, i32 %d)
  %res = call <8 x i16> @llvm.s390.verimh(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c, i32 %d)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.s390.verimf(<4 x i32>, <4 x i32>, <4 x i32>, i32)
define <4 x i32> @test_verimf(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c, i32 %d) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %d
  ; CHECK-NEXT: %res = call <4 x i32> @llvm.s390.verimf(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c, i32 %d)
  %res = call <4 x i32> @llvm.s390.verimf(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c, i32 %d)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.s390.verimg(<2 x i64>, <2 x i64>, <2 x i64>, i32)
define <2 x i64> @test_verimg(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c, i32 %d) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %d
  ; CHECK-NEXT: %res = call <2 x i64> @llvm.s390.verimg(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c, i32 %d)
  %res = call <2 x i64> @llvm.s390.verimg(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c, i32 %d)
  ret <2 x i64> %res
}

declare {<2 x i64>, i32} @llvm.s390.vftcidb(<2 x double>, i32)
define i32 @test_vftcidb(<2 x double> %a, i32 %b) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %b
  ; CHECK-NEXT: %call = call { <2 x i64>, i32 } @llvm.s390.vftcidb(<2 x double> %a, i32 %b)
  %call = call {<2 x i64>, i32} @llvm.s390.vftcidb(<2 x double> %a, i32 %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  ret i32 %res
}

declare {<4 x i32>, i32} @llvm.s390.vftcisb(<4 x float>, i32)
define i32 @test_vftcisb(<4 x float> %a, i32 %b) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %b
  ; CHECK-NEXT: %call = call { <4 x i32>, i32 } @llvm.s390.vftcisb(<4 x float> %a, i32 %b)
  %call = call {<4 x i32>, i32} @llvm.s390.vftcisb(<4 x float> %a, i32 %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  ret i32 %res
}

declare <16 x i8> @llvm.s390.vfaeb(<16 x i8>, <16 x i8>, i32)
define <16 x i8> @test_vfaeb(<16 x i8> %a, <16 x i8> %b, i32 %c) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %c
  ; CHECK-NEXT: %res = call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %a, <16 x i8> %b, i32 %c)
  %res = call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %a, <16 x i8> %b, i32 %c)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.s390.vfaezb(<16 x i8>, <16 x i8>, i32)
define <16 x i8> @test_vfaezb(<16 x i8> %a, <16 x i8> %b, i32 %c) {
  ; CHECK: immarg operand has non-immediate parameter
  ; CHECK-NEXT: i32 %c
  ; CHECK-NEXT: %res = call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %a, <16 x i8> %b, i32 %c)
  %res = call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %a, <16 x i8> %b, i32 %c)
  ret <16 x i8> %res
}
