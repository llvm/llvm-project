; Test vector intrinsics added with z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare <2 x i64> @llvm.s390.vbperm(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vmslg(<2 x i64>, <2 x i64>, <16 x i8>, i32)
declare <16 x i8> @llvm.s390.vlrl(i32, i8 *)
declare void @llvm.s390.vstrl(<16 x i8>, i32, i8 *)

declare {<4 x i32>, i32} @llvm.s390.vfcesbs(<4 x float>, <4 x float>)
declare {<4 x i32>, i32} @llvm.s390.vfchsbs(<4 x float>, <4 x float>)
declare {<4 x i32>, i32} @llvm.s390.vfchesbs(<4 x float>, <4 x float>)
declare {<4 x i32>, i32} @llvm.s390.vftcisb(<4 x float>, i32)
declare <4 x float> @llvm.s390.vfisb(<4 x float>, i32, i32)

declare <2 x double> @llvm.s390.vfmaxdb(<2 x double>, <2 x double>, i32)
declare <2 x double> @llvm.s390.vfmindb(<2 x double>, <2 x double>, i32)
declare <4 x float> @llvm.s390.vfmaxsb(<4 x float>, <4 x float>, i32)
declare <4 x float> @llvm.s390.vfminsb(<4 x float>, <4 x float>, i32)

; VBPERM.
define <2 x i64> @test_vbperm(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vbperm:
; CHECK: vbperm %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vbperm(<16 x i8> %a, <16 x i8> %b)
  ret <2 x i64> %res
}

; VMSLG with no shifts.
define <16 x i8> @test_vmslg1(<2 x i64> %a, <2 x i64> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmslg1:
; CHECK: vmslg %v24, %v24, %v26, %v28, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vmslg(<2 x i64> %a, <2 x i64> %b, <16 x i8> %c, i32 0)
  ret <16 x i8> %res
}

; VMSLG with both shifts.
define <16 x i8> @test_vmslg2(<2 x i64> %a, <2 x i64> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmslg2:
; CHECK: vmslg %v24, %v24, %v26, %v28, 12
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vmslg(<2 x i64> %a, <2 x i64> %b, <16 x i8> %c, i32 12)
  ret <16 x i8> %res
}

; VLRLR with the lowest in-range displacement.
define <16 x i8> @test_vlrlr1(i8 *%ptr, i32 %length) {
; CHECK-LABEL: test_vlrlr1:
; CHECK: vlrlr %v24, %r3, 0(%r2)
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vlrl(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRLR with the highest in-range displacement.
define <16 x i8> @test_vlrlr2(i8 *%base, i32 %length) {
; CHECK-LABEL: test_vlrlr2:
; CHECK: vlrlr %v24, %r3, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  %res = call <16 x i8> @llvm.s390.vlrl(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRLR with an out-of-range displacement.
define <16 x i8> @test_vlrlr3(i8 *%base, i32 %length) {
; CHECK-LABEL: test_vlrlr3:
; CHECK: vlrlr %v24, %r3, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  %res = call <16 x i8> @llvm.s390.vlrl(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; Check that VLRLR doesn't allow an index.
define <16 x i8> @test_vlrlr4(i8 *%base, i64 %index, i32 %length) {
; CHECK-LABEL: test_vlrlr4:
; CHECK: vlrlr %v24, %r4, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  %res = call <16 x i8> @llvm.s390.vlrl(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRL with the lowest in-range displacement.
define <16 x i8> @test_vlrl1(i8 *%ptr) {
; CHECK-LABEL: test_vlrl1:
; CHECK: vlrl %v24, 0(%r2), 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vlrl(i32 0, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRL with the highest in-range displacement.
define <16 x i8> @test_vlrl2(i8 *%base) {
; CHECK-LABEL: test_vlrl2:
; CHECK: vlrl %v24, 4095(%r2), 0
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  %res = call <16 x i8> @llvm.s390.vlrl(i32 0, i8 *%ptr)
  ret <16 x i8> %res
}

; VLRL with an out-of-range displacement.
define <16 x i8> @test_vlrl3(i8 *%base) {
; CHECK-LABEL: test_vlrl3:
; CHECK: vlrl %v24, 0({{%r[1-5]}}), 0
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  %res = call <16 x i8> @llvm.s390.vlrl(i32 0, i8 *%ptr)
  ret <16 x i8> %res
}

; Check that VLRL doesn't allow an index.
define <16 x i8> @test_vlrl4(i8 *%base, i64 %index) {
; CHECK-LABEL: test_vlrl4:
; CHECK: vlrl %v24, 0({{%r[1-5]}}), 0
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  %res = call <16 x i8> @llvm.s390.vlrl(i32 0, i8 *%ptr)
  ret <16 x i8> %res
}

; VSTRLR with the lowest in-range displacement.
define void @test_vstrlr1(<16 x i8> %vec, i8 *%ptr, i32 %length) {
; CHECK-LABEL: test_vstrlr1:
; CHECK: vstrlr %v24, %r3, 0(%r2)
; CHECK: br %r14
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; VSTRLR with the highest in-range displacement.
define void @test_vstrlr2(<16 x i8> %vec, i8 *%base, i32 %length) {
; CHECK-LABEL: test_vstrlr2:
; CHECK: vstrlr %v24, %r3, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; VSTRLR with an out-of-range displacement.
define void @test_vstrlr3(<16 x i8> %vec, i8 *%base, i32 %length) {
; CHECK-LABEL: test_vstrlr3:
; CHECK: vstrlr %v24, %r3, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; Check that VSTRLR doesn't allow an index.
define void @test_vstrlr4(<16 x i8> %vec, i8 *%base, i64 %index, i32 %length) {
; CHECK-LABEL: test_vstrlr4:
; CHECK: vstrlr %v24, %r4, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; VSTRL with the lowest in-range displacement.
define void @test_vstrl1(<16 x i8> %vec, i8 *%ptr) {
; CHECK-LABEL: test_vstrl1:
; CHECK: vstrl %v24, 0(%r2), 8
; CHECK: br %r14
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 8, i8 *%ptr)
  ret void
}

; VSTRL with the highest in-range displacement.
define void @test_vstrl2(<16 x i8> %vec, i8 *%base) {
; CHECK-LABEL: test_vstrl2:
; CHECK: vstrl %v24, 4095(%r2), 8
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 8, i8 *%ptr)
  ret void
}

; VSTRL with an out-of-range displacement.
define void @test_vstrl3(<16 x i8> %vec, i8 *%base) {
; CHECK-LABEL: test_vstrl3:
; CHECK: vstrl %v24, 0({{%r[1-5]}}), 8
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 8, i8 *%ptr)
  ret void
}

; Check that VSTRL doesn't allow an index.
define void @test_vstrl4(<16 x i8> %vec, i8 *%base, i64 %index) {
; CHECK-LABEL: test_vstrl4:
; CHECK: vstrl %v24, 0({{%r[1-5]}}), 8
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  call void @llvm.s390.vstrl(<16 x i8> %vec, i32 8, i8 *%ptr)
  ret void
}

; VFCESBS with no processing of the result.
define i32 @test_vfcesbs(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vfcesbs:
; CHECK: vfcesbs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfcesbs(<4 x float> %a,
                                                   <4 x float> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  ret i32 %res
}

; VFCESBS, returning 1 if any elements are equal (CC != 3).
define i32 @test_vfcesbs_any_bool(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vfcesbs_any_bool:
; CHECK: vfcesbs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochile %r2, 1
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfcesbs(<4 x float> %a,
                                                   <4 x float> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp ne i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VFCESBS, storing to %ptr if any elements are equal.
define <4 x i32> @test_vfcesbs_any_store(<4 x float> %a, <4 x float> %b,
                                         i32 *%ptr) {
; CHECK-LABEL: test_vfcesbs_any_store:
; CHECK-NOT: %r
; CHECK: vfcesbs %v24, %v24, %v26
; CHECK-NEXT: {{bor|bnler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfcesbs(<4 x float> %a,
                                                   <4 x float> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp ule i32 %cc, 2
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <4 x i32> %res
}

; VFCHSBS with no processing of the result.
define i32 @test_vfchsbs(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vfchsbs:
; CHECK: vfchsbs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfchsbs(<4 x float> %a,
                                                   <4 x float> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  ret i32 %res
}

; VFCHSBS, returning 1 if not all elements are higher.
define i32 @test_vfchsbs_notall_bool(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vfchsbs_notall_bool:
; CHECK: vfchsbs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochinhe %r2, 1
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfchsbs(<4 x float> %a,
                                                   <4 x float> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp sge i32 %res, 1
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VFCHSBS, storing to %ptr if not all elements are higher.
define <4 x i32> @test_vfchsbs_notall_store(<4 x float> %a, <4 x float> %b,
                                            i32 *%ptr) {
; CHECK-LABEL: test_vfchsbs_notall_store:
; CHECK-NOT: %r
; CHECK: vfchsbs %v24, %v24, %v26
; CHECK-NEXT: {{bher|ber}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfchsbs(<4 x float> %a,
                                                   <4 x float> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp ugt i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <4 x i32> %res
}

; VFCHESBS with no processing of the result.
define i32 @test_vfchesbs(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vfchesbs:
; CHECK: vfchesbs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfchesbs(<4 x float> %a,
						    <4 x float> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  ret i32 %res
}

; VFCHESBS, returning 1 if neither element is higher or equal.
define i32 @test_vfchesbs_none_bool(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vfchesbs_none_bool:
; CHECK: vfchesbs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochio %r2, 1
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfchesbs(<4 x float> %a,
						    <4 x float> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp eq i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VFCHESBS, storing to %ptr if neither element is higher or equal.
define <4 x i32> @test_vfchesbs_none_store(<4 x float> %a, <4 x float> %b,
                                           i32 *%ptr) {
; CHECK-LABEL: test_vfchesbs_none_store:
; CHECK-NOT: %r
; CHECK: vfchesbs %v24, %v24, %v26
; CHECK-NEXT: {{bnor|bler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfchesbs(<4 x float> %a,
						    <4 x float> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp uge i32 %cc, 3
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <4 x i32> %res
}

; VFTCISB with the lowest useful class selector and no processing of the result.
define i32 @test_vftcisb(<4 x float> %a) {
; CHECK-LABEL: test_vftcisb:
; CHECK: vftcisb {{%v[0-9]+}}, %v24, 1
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vftcisb(<4 x float> %a, i32 1)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  ret i32 %res
}

; VFTCISB with the highest useful class selector, returning 1 if all elements
; have the right class (CC == 0).
define i32 @test_vftcisb_all_bool(<4 x float> %a) {
; CHECK-LABEL: test_vftcisb_all_bool:
; CHECK: vftcisb {{%v[0-9]+}}, %v24, 4094
; CHECK: lhi %r2, 0
; CHECK: lochie %r2, 1
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vftcisb(<4 x float> %a, i32 4094)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp eq i32 %res, 0
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VFISB with a rounding mode not usable via standard intrinsics.
define <4 x float> @test_vfisb_0_4(<4 x float> %a) {
; CHECK-LABEL: test_vfisb_0_4:
; CHECK: vfisb %v24, %v24, 0, 4
; CHECK: br %r14
  %res = call <4 x float> @llvm.s390.vfisb(<4 x float> %a, i32 0, i32 4)
  ret <4 x float> %res
}

; VFISB with IEEE-inexact exception suppressed.
define <4 x float> @test_vfisb_4_0(<4 x float> %a) {
; CHECK-LABEL: test_vfisb_4_0:
; CHECK: vfisb %v24, %v24, 4, 0
; CHECK: br %r14
  %res = call <4 x float> @llvm.s390.vfisb(<4 x float> %a, i32 4, i32 0)
  ret <4 x float> %res
}

; VFMAXDB.
define <2 x double> @test_vfmaxdb(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfmaxdb:
; CHECK: vfmaxdb %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <2 x double> @llvm.s390.vfmaxdb(<2 x double> %a, <2 x double> %b, i32 4)
  ret <2 x double> %res
}

; VFMINDB.
define <2 x double> @test_vfmindb(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfmindb:
; CHECK: vfmindb %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <2 x double> @llvm.s390.vfmindb(<2 x double> %a, <2 x double> %b, i32 4)
  ret <2 x double> %res
}

; VFMAXSB.
define <4 x float> @test_vfmaxsb(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vfmaxsb:
; CHECK: vfmaxsb %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <4 x float> @llvm.s390.vfmaxsb(<4 x float> %a, <4 x float> %b, i32 4)
  ret <4 x float> %res
}

; VFMINSB.
define <4 x float> @test_vfminsb(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vfminsb:
; CHECK: vfminsb %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <4 x float> @llvm.s390.vfminsb(<4 x float> %a, <4 x float> %b, i32 4)
  ret <4 x float> %res
}

