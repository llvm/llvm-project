; Test vector intrinsics.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare i32 @llvm.s390.lcbb(i8 *, i32)
declare <16 x i8> @llvm.s390.vlbb(i8 *, i32)
declare <16 x i8> @llvm.s390.vll(i32, i8 *)
declare <2 x i64> @llvm.s390.vpdi(<2 x i64>, <2 x i64>, i32)
declare <16 x i8> @llvm.s390.vperm(<16 x i8>, <16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vpksh(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.s390.vpksf(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.s390.vpksg(<2 x i64>, <2 x i64>)
declare {<16 x i8>, i32} @llvm.s390.vpkshs(<8 x i16>, <8 x i16>)
declare {<8 x i16>, i32} @llvm.s390.vpksfs(<4 x i32>, <4 x i32>)
declare {<4 x i32>, i32} @llvm.s390.vpksgs(<2 x i64>, <2 x i64>)
declare <16 x i8> @llvm.s390.vpklsh(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.s390.vpklsf(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.s390.vpklsg(<2 x i64>, <2 x i64>)
declare {<16 x i8>, i32} @llvm.s390.vpklshs(<8 x i16>, <8 x i16>)
declare {<8 x i16>, i32} @llvm.s390.vpklsfs(<4 x i32>, <4 x i32>)
declare {<4 x i32>, i32} @llvm.s390.vpklsgs(<2 x i64>, <2 x i64>)
declare void @llvm.s390.vstl(<16 x i8>, i32, i8 *)
declare <8 x i16> @llvm.s390.vuphb(<16 x i8>)
declare <4 x i32> @llvm.s390.vuphh(<8 x i16>)
declare <2 x i64> @llvm.s390.vuphf(<4 x i32>)
declare <8 x i16> @llvm.s390.vuplhb(<16 x i8>)
declare <4 x i32> @llvm.s390.vuplhh(<8 x i16>)
declare <2 x i64> @llvm.s390.vuplhf(<4 x i32>)
declare <8 x i16> @llvm.s390.vuplb(<16 x i8>)
declare <4 x i32> @llvm.s390.vuplhw(<8 x i16>)
declare <2 x i64> @llvm.s390.vuplf(<4 x i32>)
declare <8 x i16> @llvm.s390.vupllb(<16 x i8>)
declare <4 x i32> @llvm.s390.vupllh(<8 x i16>)
declare <2 x i64> @llvm.s390.vupllf(<4 x i32>)
declare <16 x i8> @llvm.s390.vaccb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vacch(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vaccf(<4 x i32>, <4 x i32>)
declare <2 x i64> @llvm.s390.vaccg(<2 x i64>, <2 x i64>)
declare <16 x i8> @llvm.s390.vaq(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vacq(<16 x i8>, <16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vaccq(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vacccq(<16 x i8>, <16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vavgb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vavgh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vavgf(<4 x i32>, <4 x i32>)
declare <2 x i64> @llvm.s390.vavgg(<2 x i64>, <2 x i64>)
declare <16 x i8> @llvm.s390.vavglb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vavglh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vavglf(<4 x i32>, <4 x i32>)
declare <2 x i64> @llvm.s390.vavglg(<2 x i64>, <2 x i64>)
declare <4 x i32> @llvm.s390.vcksm(<4 x i32>, <4 x i32>)
declare <8 x i16> @llvm.s390.vgfmb(<16 x i8>, <16 x i8>)
declare <4 x i32> @llvm.s390.vgfmh(<8 x i16>, <8 x i16>)
declare <2 x i64> @llvm.s390.vgfmf(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.vgfmg(<2 x i64>, <2 x i64>)
declare <8 x i16> @llvm.s390.vgfmab(<16 x i8>, <16 x i8>, <8 x i16>)
declare <4 x i32> @llvm.s390.vgfmah(<8 x i16>, <8 x i16>, <4 x i32>)
declare <2 x i64> @llvm.s390.vgfmaf(<4 x i32>, <4 x i32>, <2 x i64>)
declare <16 x i8> @llvm.s390.vgfmag(<2 x i64>, <2 x i64>, <16 x i8>)
declare <16 x i8> @llvm.s390.vmahb(<16 x i8>, <16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vmahh(<8 x i16>, <8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vmahf(<4 x i32>, <4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.vmalhb(<16 x i8>, <16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vmalhh(<8 x i16>, <8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vmalhf(<4 x i32>, <4 x i32>, <4 x i32>)
declare <8 x i16> @llvm.s390.vmaeb(<16 x i8>, <16 x i8>, <8 x i16>)
declare <4 x i32> @llvm.s390.vmaeh(<8 x i16>, <8 x i16>, <4 x i32>)
declare <2 x i64> @llvm.s390.vmaef(<4 x i32>, <4 x i32>, <2 x i64>)
declare <8 x i16> @llvm.s390.vmaleb(<16 x i8>, <16 x i8>, <8 x i16>)
declare <4 x i32> @llvm.s390.vmaleh(<8 x i16>, <8 x i16>, <4 x i32>)
declare <2 x i64> @llvm.s390.vmalef(<4 x i32>, <4 x i32>, <2 x i64>)
declare <8 x i16> @llvm.s390.vmaob(<16 x i8>, <16 x i8>, <8 x i16>)
declare <4 x i32> @llvm.s390.vmaoh(<8 x i16>, <8 x i16>, <4 x i32>)
declare <2 x i64> @llvm.s390.vmaof(<4 x i32>, <4 x i32>, <2 x i64>)
declare <8 x i16> @llvm.s390.vmalob(<16 x i8>, <16 x i8>, <8 x i16>)
declare <4 x i32> @llvm.s390.vmaloh(<8 x i16>, <8 x i16>, <4 x i32>)
declare <2 x i64> @llvm.s390.vmalof(<4 x i32>, <4 x i32>, <2 x i64>)
declare <16 x i8> @llvm.s390.vmhb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vmhh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vmhf(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.vmlhb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vmlhh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vmlhf(<4 x i32>, <4 x i32>)
declare <8 x i16> @llvm.s390.vmeb(<16 x i8>, <16 x i8>)
declare <4 x i32> @llvm.s390.vmeh(<8 x i16>, <8 x i16>)
declare <2 x i64> @llvm.s390.vmef(<4 x i32>, <4 x i32>)
declare <8 x i16> @llvm.s390.vmleb(<16 x i8>, <16 x i8>)
declare <4 x i32> @llvm.s390.vmleh(<8 x i16>, <8 x i16>)
declare <2 x i64> @llvm.s390.vmlef(<4 x i32>, <4 x i32>)
declare <8 x i16> @llvm.s390.vmob(<16 x i8>, <16 x i8>)
declare <4 x i32> @llvm.s390.vmoh(<8 x i16>, <8 x i16>)
declare <2 x i64> @llvm.s390.vmof(<4 x i32>, <4 x i32>)
declare <8 x i16> @llvm.s390.vmlob(<16 x i8>, <16 x i8>)
declare <4 x i32> @llvm.s390.vmloh(<8 x i16>, <8 x i16>)
declare <2 x i64> @llvm.s390.vmlof(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.verllvb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.verllvh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.verllvf(<4 x i32>, <4 x i32>)
declare <2 x i64> @llvm.s390.verllvg(<2 x i64>, <2 x i64>)
declare <16 x i8> @llvm.s390.verllb(<16 x i8>, i32)
declare <8 x i16> @llvm.s390.verllh(<8 x i16>, i32)
declare <4 x i32> @llvm.s390.verllf(<4 x i32>, i32)
declare <2 x i64> @llvm.s390.verllg(<2 x i64>, i32)
declare <16 x i8> @llvm.s390.verimb(<16 x i8>, <16 x i8>, <16 x i8>, i32)
declare <8 x i16> @llvm.s390.verimh(<8 x i16>, <8 x i16>, <8 x i16>, i32)
declare <4 x i32> @llvm.s390.verimf(<4 x i32>, <4 x i32>, <4 x i32>, i32)
declare <2 x i64> @llvm.s390.verimg(<2 x i64>, <2 x i64>, <2 x i64>, i32)
declare <16 x i8> @llvm.s390.vsl(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vslb(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vsra(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vsrab(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vsrl(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vsrlb(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vsldb(<16 x i8>, <16 x i8>, i32)
declare <16 x i8> @llvm.s390.vscbib(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vscbih(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vscbif(<4 x i32>, <4 x i32>)
declare <2 x i64> @llvm.s390.vscbig(<2 x i64>, <2 x i64>)
declare <16 x i8> @llvm.s390.vsq(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vsbiq(<16 x i8>, <16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vscbiq(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.s390.vsbcbiq(<16 x i8>, <16 x i8>, <16 x i8>)
declare <4 x i32> @llvm.s390.vsumb(<16 x i8>, <16 x i8>)
declare <4 x i32> @llvm.s390.vsumh(<8 x i16>, <8 x i16>)
declare <2 x i64> @llvm.s390.vsumgh(<8 x i16>, <8 x i16>)
declare <2 x i64> @llvm.s390.vsumgf(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.vsumqf(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.vsumqg(<2 x i64>, <2 x i64>)
declare i32 @llvm.s390.vtm(<16 x i8>, <16 x i8>)
declare {<16 x i8>, i32} @llvm.s390.vceqbs(<16 x i8>, <16 x i8>)
declare {<8 x i16>, i32} @llvm.s390.vceqhs(<8 x i16>, <8 x i16>)
declare {<4 x i32>, i32} @llvm.s390.vceqfs(<4 x i32>, <4 x i32>)
declare {<2 x i64>, i32} @llvm.s390.vceqgs(<2 x i64>, <2 x i64>)
declare {<16 x i8>, i32} @llvm.s390.vchbs(<16 x i8>, <16 x i8>)
declare {<8 x i16>, i32} @llvm.s390.vchhs(<8 x i16>, <8 x i16>)
declare {<4 x i32>, i32} @llvm.s390.vchfs(<4 x i32>, <4 x i32>)
declare {<2 x i64>, i32} @llvm.s390.vchgs(<2 x i64>, <2 x i64>)
declare {<16 x i8>, i32} @llvm.s390.vchlbs(<16 x i8>, <16 x i8>)
declare {<8 x i16>, i32} @llvm.s390.vchlhs(<8 x i16>, <8 x i16>)
declare {<4 x i32>, i32} @llvm.s390.vchlfs(<4 x i32>, <4 x i32>)
declare {<2 x i64>, i32} @llvm.s390.vchlgs(<2 x i64>, <2 x i64>)
declare <16 x i8> @llvm.s390.vfaeb(<16 x i8>, <16 x i8>, i32)
declare <8 x i16> @llvm.s390.vfaeh(<8 x i16>, <8 x i16>, i32)
declare <4 x i32> @llvm.s390.vfaef(<4 x i32>, <4 x i32>, i32)
declare {<16 x i8>, i32} @llvm.s390.vfaebs(<16 x i8>, <16 x i8>, i32)
declare {<8 x i16>, i32} @llvm.s390.vfaehs(<8 x i16>, <8 x i16>, i32)
declare {<4 x i32>, i32} @llvm.s390.vfaefs(<4 x i32>, <4 x i32>, i32)
declare <16 x i8> @llvm.s390.vfaezb(<16 x i8>, <16 x i8>, i32)
declare <8 x i16> @llvm.s390.vfaezh(<8 x i16>, <8 x i16>, i32)
declare <4 x i32> @llvm.s390.vfaezf(<4 x i32>, <4 x i32>, i32)
declare {<16 x i8>, i32} @llvm.s390.vfaezbs(<16 x i8>, <16 x i8>, i32)
declare {<8 x i16>, i32} @llvm.s390.vfaezhs(<8 x i16>, <8 x i16>, i32)
declare {<4 x i32>, i32} @llvm.s390.vfaezfs(<4 x i32>, <4 x i32>, i32)
declare <16 x i8> @llvm.s390.vfeeb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vfeeh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vfeef(<4 x i32>, <4 x i32>)
declare {<16 x i8>, i32} @llvm.s390.vfeebs(<16 x i8>, <16 x i8>)
declare {<8 x i16>, i32} @llvm.s390.vfeehs(<8 x i16>, <8 x i16>)
declare {<4 x i32>, i32} @llvm.s390.vfeefs(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.vfeezb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vfeezh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vfeezf(<4 x i32>, <4 x i32>)
declare {<16 x i8>, i32} @llvm.s390.vfeezbs(<16 x i8>, <16 x i8>)
declare {<8 x i16>, i32} @llvm.s390.vfeezhs(<8 x i16>, <8 x i16>)
declare {<4 x i32>, i32} @llvm.s390.vfeezfs(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.vfeneb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vfeneh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vfenef(<4 x i32>, <4 x i32>)
declare {<16 x i8>, i32} @llvm.s390.vfenebs(<16 x i8>, <16 x i8>)
declare {<8 x i16>, i32} @llvm.s390.vfenehs(<8 x i16>, <8 x i16>)
declare {<4 x i32>, i32} @llvm.s390.vfenefs(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.vfenezb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.s390.vfenezh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.s390.vfenezf(<4 x i32>, <4 x i32>)
declare {<16 x i8>, i32} @llvm.s390.vfenezbs(<16 x i8>, <16 x i8>)
declare {<8 x i16>, i32} @llvm.s390.vfenezhs(<8 x i16>, <8 x i16>)
declare {<4 x i32>, i32} @llvm.s390.vfenezfs(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.s390.vistrb(<16 x i8>)
declare <8 x i16> @llvm.s390.vistrh(<8 x i16>)
declare <4 x i32> @llvm.s390.vistrf(<4 x i32>)
declare {<16 x i8>, i32} @llvm.s390.vistrbs(<16 x i8>)
declare {<8 x i16>, i32} @llvm.s390.vistrhs(<8 x i16>)
declare {<4 x i32>, i32} @llvm.s390.vistrfs(<4 x i32>)
declare <16 x i8> @llvm.s390.vstrcb(<16 x i8>, <16 x i8>, <16 x i8>, i32)
declare <8 x i16> @llvm.s390.vstrch(<8 x i16>, <8 x i16>, <8 x i16>, i32)
declare <4 x i32> @llvm.s390.vstrcf(<4 x i32>, <4 x i32>, <4 x i32>, i32)
declare {<16 x i8>, i32} @llvm.s390.vstrcbs(<16 x i8>, <16 x i8>, <16 x i8>,
                                            i32)
declare {<8 x i16>, i32} @llvm.s390.vstrchs(<8 x i16>, <8 x i16>, <8 x i16>,
                                            i32)
declare {<4 x i32>, i32} @llvm.s390.vstrcfs(<4 x i32>, <4 x i32>, <4 x i32>,
                                            i32)
declare <16 x i8> @llvm.s390.vstrczb(<16 x i8>, <16 x i8>, <16 x i8>, i32)
declare <8 x i16> @llvm.s390.vstrczh(<8 x i16>, <8 x i16>, <8 x i16>, i32)
declare <4 x i32> @llvm.s390.vstrczf(<4 x i32>, <4 x i32>, <4 x i32>, i32)
declare {<16 x i8>, i32} @llvm.s390.vstrczbs(<16 x i8>, <16 x i8>, <16 x i8>,
                                             i32)
declare {<8 x i16>, i32} @llvm.s390.vstrczhs(<8 x i16>, <8 x i16>, <8 x i16>,
                                             i32)
declare {<4 x i32>, i32} @llvm.s390.vstrczfs(<4 x i32>, <4 x i32>, <4 x i32>,
                                             i32)
declare {<2 x i64>, i32} @llvm.s390.vfcedbs(<2 x double>, <2 x double>)
declare {<2 x i64>, i32} @llvm.s390.vfchdbs(<2 x double>, <2 x double>)
declare {<2 x i64>, i32} @llvm.s390.vfchedbs(<2 x double>, <2 x double>)
declare {<2 x i64>, i32} @llvm.s390.vftcidb(<2 x double>, i32)
declare <2 x double> @llvm.s390.vfidb(<2 x double>, i32, i32)

; LCBB with the lowest M3 operand.
define i32 @test_lcbb1(i8 *%ptr) {
; CHECK-LABEL: test_lcbb1:
; CHECK: lcbb %r2, 0(%r2), 0
; CHECK: br %r14
  %res = call i32 @llvm.s390.lcbb(i8 *%ptr, i32 0)
  ret i32 %res
}

; LCBB with the highest M3 operand.
define i32 @test_lcbb2(i8 *%ptr) {
; CHECK-LABEL: test_lcbb2:
; CHECK: lcbb %r2, 0(%r2), 15
; CHECK: br %r14
  %res = call i32 @llvm.s390.lcbb(i8 *%ptr, i32 15)
  ret i32 %res
}

; LCBB with a displacement and index.
define i32 @test_lcbb3(i8 *%base, i64 %index) {
; CHECK-LABEL: test_lcbb3:
; CHECK: lcbb %r2, 4095({{%r2,%r3|%r3,%r2}}), 4
; CHECK: br %r14
  %add = add i64 %index, 4095
  %ptr = getelementptr i8, i8 *%base, i64 %add
  %res = call i32 @llvm.s390.lcbb(i8 *%ptr, i32 4)
  ret i32 %res
}

; LCBB with an out-of-range displacement.
define i32 @test_lcbb4(i8 *%base) {
; CHECK-LABEL: test_lcbb4:
; CHECK: lcbb %r2, 0({{%r[1-5]}}), 5
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  %res = call i32 @llvm.s390.lcbb(i8 *%ptr, i32 5)
  ret i32 %res
}

; VLBB with the lowest M3 operand.
define <16 x i8> @test_vlbb1(i8 *%ptr) {
; CHECK-LABEL: test_vlbb1:
; CHECK: vlbb %v24, 0(%r2), 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vlbb(i8 *%ptr, i32 0)
  ret <16 x i8> %res
}

; VLBB with the highest M3 operand.
define <16 x i8> @test_vlbb2(i8 *%ptr) {
; CHECK-LABEL: test_vlbb2:
; CHECK: vlbb %v24, 0(%r2), 15
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vlbb(i8 *%ptr, i32 15)
  ret <16 x i8> %res
}

; VLBB with a displacement and index.
define <16 x i8> @test_vlbb3(i8 *%base, i64 %index) {
; CHECK-LABEL: test_vlbb3:
; CHECK: vlbb %v24, 4095({{%r2,%r3|%r3,%r2}}), 4
; CHECK: br %r14
  %add = add i64 %index, 4095
  %ptr = getelementptr i8, i8 *%base, i64 %add
  %res = call <16 x i8> @llvm.s390.vlbb(i8 *%ptr, i32 4)
  ret <16 x i8> %res
}

; VLBB with an out-of-range displacement.
define <16 x i8> @test_vlbb4(i8 *%base) {
; CHECK-LABEL: test_vlbb4:
; CHECK: vlbb %v24, 0({{%r[1-5]}}), 5
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  %res = call <16 x i8> @llvm.s390.vlbb(i8 *%ptr, i32 5)
  ret <16 x i8> %res
}

; VLL with the lowest in-range displacement.
define <16 x i8> @test_vll1(i8 *%ptr, i32 %length) {
; CHECK-LABEL: test_vll1:
; CHECK: vll %v24, %r3, 0(%r2)
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vll(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; VLL with the highest in-range displacement.
define <16 x i8> @test_vll2(i8 *%base, i32 %length) {
; CHECK-LABEL: test_vll2:
; CHECK: vll %v24, %r3, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  %res = call <16 x i8> @llvm.s390.vll(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; VLL with an out-of-range displacementa.
define <16 x i8> @test_vll3(i8 *%base, i32 %length) {
; CHECK-LABEL: test_vll3:
; CHECK: vll %v24, %r3, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  %res = call <16 x i8> @llvm.s390.vll(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; Check that VLL doesn't allow an index.
define <16 x i8> @test_vll4(i8 *%base, i64 %index, i32 %length) {
; CHECK-LABEL: test_vll4:
; CHECK: vll %v24, %r4, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  %res = call <16 x i8> @llvm.s390.vll(i32 %length, i8 *%ptr)
  ret <16 x i8> %res
}

; VLL with length >= 15 should become VL.
define <16 x i8> @test_vll5(i8 *%ptr) {
; CHECK-LABEL: test_vll5:
; CHECK: vl %v24, 0({{%r[1-5]}})
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vll(i32 15, i8 *%ptr)
  ret <16 x i8> %res
}

; VPDI taking element 0 from each half.
define <2 x i64> @test_vpdi1(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vpdi1:
; CHECK: vpdi %v24, %v24, %v26, 0
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vpdi(<2 x i64> %a, <2 x i64> %b, i32 0)
  ret <2 x i64> %res
}

; VPDI taking element 1 from each half.
define <2 x i64> @test_vpdi2(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vpdi2:
; CHECK: vpdi %v24, %v24, %v26, 5
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vpdi(<2 x i64> %a, <2 x i64> %b, i32 5)
  ret <2 x i64> %res
}

; VPERM.
define <16 x i8> @test_vperm(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vperm:
; CHECK: vperm %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vperm(<16 x i8> %a, <16 x i8> %b,
                                         <16 x i8> %c)
  ret <16 x i8> %res
}

; VPKSH.
define <16 x i8> @test_vpksh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vpksh:
; CHECK: vpksh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vpksh(<8 x i16> %a, <8 x i16> %b)
  ret <16 x i8> %res
}

; VPKSF.
define <8 x i16> @test_vpksf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vpksf:
; CHECK: vpksf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vpksf(<4 x i32> %a, <4 x i32> %b)
  ret <8 x i16> %res
}

; VPKSG.
define <4 x i32> @test_vpksg(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vpksg:
; CHECK: vpksg %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vpksg(<2 x i64> %a, <2 x i64> %b)
  ret <4 x i32> %res
}

; VPKSHS with no processing of the result.
define <16 x i8> @test_vpkshs(<8 x i16> %a, <8 x i16> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vpkshs:
; CHECK: vpkshs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpkshs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VPKSHS, storing to %ptr if all values were saturated.
define <16 x i8> @test_vpkshs_all_store(<8 x i16> %a, <8 x i16> %b, i32 *%ptr) {
; CHECK-LABEL: test_vpkshs_all_store:
; CHECK: vpkshs %v24, %v24, %v26
; CHECK-NEXT: {{bnor|bler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpkshs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  %cmp = icmp uge i32 %cc, 3
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <16 x i8> %res
}

; VPKSFS with no processing of the result.
define <8 x i16> @test_vpksfs(<4 x i32> %a, <4 x i32> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vpksfs:
; CHECK: vpksfs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpksfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VPKSFS, storing to %ptr if any values were saturated.
define <8 x i16> @test_vpksfs_any_store(<4 x i32> %a, <4 x i32> %b, i32 *%ptr) {
; CHECK-LABEL: test_vpksfs_any_store:
; CHECK: vpksfs %v24, %v24, %v26
; CHECK-NEXT: {{bher|ber}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpksfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  %cmp = icmp ugt i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <8 x i16> %res
}

; VPKSGS with no processing of the result.
define <4 x i32> @test_vpksgs(<2 x i64> %a, <2 x i64> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vpksgs:
; CHECK: vpksgs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpksgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VPKSGS, storing to %ptr if no elements were saturated
define <4 x i32> @test_vpksgs_none_store(<2 x i64> %a, <2 x i64> %b,
                                         i32 *%ptr) {
; CHECK-LABEL: test_vpksgs_none_store:
; CHECK: vpksgs %v24, %v24, %v26
; CHECK-NEXT: {{bnher|bner}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpksgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp sle i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <4 x i32> %res
}

; VPKLSH.
define <16 x i8> @test_vpklsh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vpklsh:
; CHECK: vpklsh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vpklsh(<8 x i16> %a, <8 x i16> %b)
  ret <16 x i8> %res
}

; VPKLSF.
define <8 x i16> @test_vpklsf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vpklsf:
; CHECK: vpklsf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vpklsf(<4 x i32> %a, <4 x i32> %b)
  ret <8 x i16> %res
}

; VPKLSG.
define <4 x i32> @test_vpklsg(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vpklsg:
; CHECK: vpklsg %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vpklsg(<2 x i64> %a, <2 x i64> %b)
  ret <4 x i32> %res
}

; VPKLSHS with no processing of the result.
define <16 x i8> @test_vpklshs(<8 x i16> %a, <8 x i16> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vpklshs:
; CHECK: vpklshs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpklshs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VPKLSHS, storing to %ptr if all values were saturated.
define <16 x i8> @test_vpklshs_all_store(<8 x i16> %a, <8 x i16> %b,
                                         i32 *%ptr) {
; CHECK-LABEL: test_vpklshs_all_store:
; CHECK: vpklshs %v24, %v24, %v26
; CHECK-NEXT: {{bnor|bler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vpklshs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  %cmp = icmp eq i32 %cc, 3
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <16 x i8> %res
}

; VPKLSFS with no processing of the result.
define <8 x i16> @test_vpklsfs(<4 x i32> %a, <4 x i32> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vpklsfs:
; CHECK: vpklsfs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpklsfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VPKLSFS, storing to %ptr if any values were saturated.
define <8 x i16> @test_vpklsfs_any_store(<4 x i32> %a, <4 x i32> %b,
                                         i32 *%ptr) {
; CHECK-LABEL: test_vpklsfs_any_store:
; CHECK: vpklsfs %v24, %v24, %v26
; CHECK-NEXT: {{bher|ber}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vpklsfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  %cmp = icmp ne i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <8 x i16> %res
}

; VPKLSGS with no processing of the result.
define <4 x i32> @test_vpklsgs(<2 x i64> %a, <2 x i64> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vpklsgs:
; CHECK: vpklsgs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpklsgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VPKLSGS, storing to %ptr if no elements were saturated
define <4 x i32> @test_vpklsgs_none_store(<2 x i64> %a, <2 x i64> %b,
                                          i32 *%ptr) {
; CHECK-LABEL: test_vpklsgs_none_store:
; CHECK: vpklsgs %v24, %v24, %v26
; CHECK-NEXT: {{bnher|bner}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vpklsgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp eq i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <4 x i32> %res
}

; VSTL with the lowest in-range displacement.
define void @test_vstl1(<16 x i8> %vec, i8 *%ptr, i32 %length) {
; CHECK-LABEL: test_vstl1:
; CHECK: vstl %v24, %r3, 0(%r2)
; CHECK: br %r14
  call void @llvm.s390.vstl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; VSTL with the highest in-range displacement.
define void @test_vstl2(<16 x i8> %vec, i8 *%base, i32 %length) {
; CHECK-LABEL: test_vstl2:
; CHECK: vstl %v24, %r3, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  call void @llvm.s390.vstl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; VSTL with an out-of-range displacement.
define void @test_vstl3(<16 x i8> %vec, i8 *%base, i32 %length) {
; CHECK-LABEL: test_vstl3:
; CHECK: vstl %v24, %r3, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  call void @llvm.s390.vstl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; Check that VSTL doesn't allow an index.
define void @test_vstl4(<16 x i8> %vec, i8 *%base, i64 %index, i32 %length) {
; CHECK-LABEL: test_vstl4:
; CHECK: vstl %v24, %r4, 0({{%r[1-5]}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  call void @llvm.s390.vstl(<16 x i8> %vec, i32 %length, i8 *%ptr)
  ret void
}

; VSTL with length >= 15 should become VST.
define void @test_vstl5(<16 x i8> %vec, i8 *%ptr) {
; CHECK-LABEL: test_vstl5:
; CHECK: vst %v24, 0({{%r[1-5]}})
; CHECK: br %r14
  call void @llvm.s390.vstl(<16 x i8> %vec, i32 15, i8 *%ptr)
  ret void
}

; VUPHB.
define <8 x i16> @test_vuphb(<16 x i8> %a) {
; CHECK-LABEL: test_vuphb:
; CHECK: vuphb %v24, %v24
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vuphb(<16 x i8> %a)
  ret <8 x i16> %res
}

; VUPHH.
define <4 x i32> @test_vuphh(<8 x i16> %a) {
; CHECK-LABEL: test_vuphh:
; CHECK: vuphh %v24, %v24
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vuphh(<8 x i16> %a)
  ret <4 x i32> %res
}

; VUPHF.
define <2 x i64> @test_vuphf(<4 x i32> %a) {
; CHECK-LABEL: test_vuphf:
; CHECK: vuphf %v24, %v24
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vuphf(<4 x i32> %a)
  ret <2 x i64> %res
}

; VUPLHB.
define <8 x i16> @test_vuplhb(<16 x i8> %a) {
; CHECK-LABEL: test_vuplhb:
; CHECK: vuplhb %v24, %v24
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vuplhb(<16 x i8> %a)
  ret <8 x i16> %res
}

; VUPLHH.
define <4 x i32> @test_vuplhh(<8 x i16> %a) {
; CHECK-LABEL: test_vuplhh:
; CHECK: vuplhh %v24, %v24
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vuplhh(<8 x i16> %a)
  ret <4 x i32> %res
}

; VUPLHF.
define <2 x i64> @test_vuplhf(<4 x i32> %a) {
; CHECK-LABEL: test_vuplhf:
; CHECK: vuplhf %v24, %v24
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vuplhf(<4 x i32> %a)
  ret <2 x i64> %res
}

; VUPLB.
define <8 x i16> @test_vuplb(<16 x i8> %a) {
; CHECK-LABEL: test_vuplb:
; CHECK: vuplb %v24, %v24
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vuplb(<16 x i8> %a)
  ret <8 x i16> %res
}

; VUPLHW.
define <4 x i32> @test_vuplhw(<8 x i16> %a) {
; CHECK-LABEL: test_vuplhw:
; CHECK: vuplhw %v24, %v24
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vuplhw(<8 x i16> %a)
  ret <4 x i32> %res
}

; VUPLF.
define <2 x i64> @test_vuplf(<4 x i32> %a) {
; CHECK-LABEL: test_vuplf:
; CHECK: vuplf %v24, %v24
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vuplf(<4 x i32> %a)
  ret <2 x i64> %res
}

; VUPLLB.
define <8 x i16> @test_vupllb(<16 x i8> %a) {
; CHECK-LABEL: test_vupllb:
; CHECK: vupllb %v24, %v24
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vupllb(<16 x i8> %a)
  ret <8 x i16> %res
}

; VUPLLH.
define <4 x i32> @test_vupllh(<8 x i16> %a) {
; CHECK-LABEL: test_vupllh:
; CHECK: vupllh %v24, %v24
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vupllh(<8 x i16> %a)
  ret <4 x i32> %res
}

; VUPLLF.
define <2 x i64> @test_vupllf(<4 x i32> %a) {
; CHECK-LABEL: test_vupllf:
; CHECK: vupllf %v24, %v24
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vupllf(<4 x i32> %a)
  ret <2 x i64> %res
}

; VACCB.
define <16 x i8> @test_vaccb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vaccb:
; CHECK: vaccb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vaccb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VACCH.
define <8 x i16> @test_vacch(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vacch:
; CHECK: vacch %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vacch(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VACCF.
define <4 x i32> @test_vaccf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vaccf:
; CHECK: vaccf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vaccf(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VACCG.
define <2 x i64> @test_vaccg(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vaccg:
; CHECK: vaccg %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vaccg(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %res
}

; VAQ.
define <16 x i8> @test_vaq(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vaq:
; CHECK: vaq %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vaq(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VACQ.
define <16 x i8> @test_vacq(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vacq:
; CHECK: vacq %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vacq(<16 x i8> %a, <16 x i8> %b,
                                        <16 x i8> %c)
  ret <16 x i8> %res
}

; VACCQ.
define <16 x i8> @test_vaccq(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vaccq:
; CHECK: vaccq %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vaccq(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VACCCQ.
define <16 x i8> @test_vacccq(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vacccq:
; CHECK: vacccq %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vacccq(<16 x i8> %a, <16 x i8> %b,
                                          <16 x i8> %c)
  ret <16 x i8> %res
}

; VAVGB.
define <16 x i8> @test_vavgb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vavgb:
; CHECK: vavgb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vavgb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VAVGH.
define <8 x i16> @test_vavgh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vavgh:
; CHECK: vavgh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vavgh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VAVGF.
define <4 x i32> @test_vavgf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vavgf:
; CHECK: vavgf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vavgf(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VAVGG.
define <2 x i64> @test_vavgg(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vavgg:
; CHECK: vavgg %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vavgg(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %res
}

; VAVGLB.
define <16 x i8> @test_vavglb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vavglb:
; CHECK: vavglb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vavglb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VAVGLH.
define <8 x i16> @test_vavglh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vavglh:
; CHECK: vavglh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vavglh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VAVGLF.
define <4 x i32> @test_vavglf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vavglf:
; CHECK: vavglf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vavglf(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VAVGLG.
define <2 x i64> @test_vavglg(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vavglg:
; CHECK: vavglg %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vavglg(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %res
}

; VCKSM.
define <4 x i32> @test_vcksm(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vcksm:
; CHECK: vcksm %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vcksm(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VGFMB.
define <8 x i16> @test_vgfmb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vgfmb:
; CHECK: vgfmb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vgfmb(<16 x i8> %a, <16 x i8> %b)
  ret <8 x i16> %res
}

; VGFMH.
define <4 x i32> @test_vgfmh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vgfmh:
; CHECK: vgfmh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vgfmh(<8 x i16> %a, <8 x i16> %b)
  ret <4 x i32> %res
}

; VGFMF.
define <2 x i64> @test_vgfmf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vgfmf:
; CHECK: vgfmf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vgfmf(<4 x i32> %a, <4 x i32> %b)
  ret <2 x i64> %res
}

; VGFMG.
define <16 x i8> @test_vgfmg(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vgfmg:
; CHECK: vgfmg %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vgfmg(<2 x i64> %a, <2 x i64> %b)
  ret <16 x i8> %res
}

; VGFMAB.
define <8 x i16> @test_vgfmab(<16 x i8> %a, <16 x i8> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vgfmab:
; CHECK: vgfmab %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vgfmab(<16 x i8> %a, <16 x i8> %b,
                                          <8 x i16> %c)
  ret <8 x i16> %res
}

; VGFMAH.
define <4 x i32> @test_vgfmah(<8 x i16> %a, <8 x i16> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vgfmah:
; CHECK: vgfmah %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vgfmah(<8 x i16> %a, <8 x i16> %b,
                                          <4 x i32> %c)
  ret <4 x i32> %res
}

; VGFMAF.
define <2 x i64> @test_vgfmaf(<4 x i32> %a, <4 x i32> %b, <2 x i64> %c) {
; CHECK-LABEL: test_vgfmaf:
; CHECK: vgfmaf %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vgfmaf(<4 x i32> %a, <4 x i32> %b,
                                          <2 x i64> %c)
  ret <2 x i64> %res
}

; VGFMAG.
define <16 x i8> @test_vgfmag(<2 x i64> %a, <2 x i64> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vgfmag:
; CHECK: vgfmag %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vgfmag(<2 x i64> %a, <2 x i64> %b,
                                          <16 x i8> %c)
  ret <16 x i8> %res
}

; VMAHB.
define <16 x i8> @test_vmahb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmahb:
; CHECK: vmahb %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vmahb(<16 x i8> %a, <16 x i8> %b,
                                         <16 x i8> %c)
  ret <16 x i8> %res
}

; VMAHH.
define <8 x i16> @test_vmahh(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmahh:
; CHECK: vmahh %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmahh(<8 x i16> %a, <8 x i16> %b,
                                         <8 x i16> %c)
  ret <8 x i16> %res
}

; VMAHF.
define <4 x i32> @test_vmahf(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmahf:
; CHECK: vmahf %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmahf(<4 x i32> %a, <4 x i32> %b,
                                         <4 x i32> %c)
  ret <4 x i32> %res
}

; VMALHB.
define <16 x i8> @test_vmalhb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmalhb:
; CHECK: vmalhb %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vmalhb(<16 x i8> %a, <16 x i8> %b,
                                          <16 x i8> %c)
  ret <16 x i8> %res
}

; VMALHH.
define <8 x i16> @test_vmalhh(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmalhh:
; CHECK: vmalhh %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmalhh(<8 x i16> %a, <8 x i16> %b,
                                          <8 x i16> %c)
  ret <8 x i16> %res
}

; VMALHF.
define <4 x i32> @test_vmalhf(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmalhf:
; CHECK: vmalhf %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmalhf(<4 x i32> %a, <4 x i32> %b,
                                          <4 x i32> %c)
  ret <4 x i32> %res
}

; VMAEB.
define <8 x i16> @test_vmaeb(<16 x i8> %a, <16 x i8> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmaeb:
; CHECK: vmaeb %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmaeb(<16 x i8> %a, <16 x i8> %b,
                                         <8 x i16> %c)
  ret <8 x i16> %res
}

; VMAEH.
define <4 x i32> @test_vmaeh(<8 x i16> %a, <8 x i16> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmaeh:
; CHECK: vmaeh %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmaeh(<8 x i16> %a, <8 x i16> %b,
                                         <4 x i32> %c)
  ret <4 x i32> %res
}

; VMAEF.
define <2 x i64> @test_vmaef(<4 x i32> %a, <4 x i32> %b, <2 x i64> %c) {
; CHECK-LABEL: test_vmaef:
; CHECK: vmaef %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vmaef(<4 x i32> %a, <4 x i32> %b,
                                         <2 x i64> %c)
  ret <2 x i64> %res
}

; VMALEB.
define <8 x i16> @test_vmaleb(<16 x i8> %a, <16 x i8> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmaleb:
; CHECK: vmaleb %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmaleb(<16 x i8> %a, <16 x i8> %b,
                                          <8 x i16> %c)
  ret <8 x i16> %res
}

; VMALEH.
define <4 x i32> @test_vmaleh(<8 x i16> %a, <8 x i16> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmaleh:
; CHECK: vmaleh %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmaleh(<8 x i16> %a, <8 x i16> %b,
                                          <4 x i32> %c)
  ret <4 x i32> %res
}

; VMALEF.
define <2 x i64> @test_vmalef(<4 x i32> %a, <4 x i32> %b, <2 x i64> %c) {
; CHECK-LABEL: test_vmalef:
; CHECK: vmalef %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vmalef(<4 x i32> %a, <4 x i32> %b,
                                          <2 x i64> %c)
  ret <2 x i64> %res
}

; VMAOB.
define <8 x i16> @test_vmaob(<16 x i8> %a, <16 x i8> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmaob:
; CHECK: vmaob %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmaob(<16 x i8> %a, <16 x i8> %b,
                                         <8 x i16> %c)
  ret <8 x i16> %res
}

; VMAOH.
define <4 x i32> @test_vmaoh(<8 x i16> %a, <8 x i16> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmaoh:
; CHECK: vmaoh %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmaoh(<8 x i16> %a, <8 x i16> %b,
                                         <4 x i32> %c)
  ret <4 x i32> %res
}

; VMAOF.
define <2 x i64> @test_vmaof(<4 x i32> %a, <4 x i32> %b, <2 x i64> %c) {
; CHECK-LABEL: test_vmaof:
; CHECK: vmaof %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vmaof(<4 x i32> %a, <4 x i32> %b,
                                         <2 x i64> %c)
  ret <2 x i64> %res
}

; VMALOB.
define <8 x i16> @test_vmalob(<16 x i8> %a, <16 x i8> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmalob:
; CHECK: vmalob %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmalob(<16 x i8> %a, <16 x i8> %b,
                                          <8 x i16> %c)
  ret <8 x i16> %res
}

; VMALOH.
define <4 x i32> @test_vmaloh(<8 x i16> %a, <8 x i16> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmaloh:
; CHECK: vmaloh %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmaloh(<8 x i16> %a, <8 x i16> %b,
                                          <4 x i32> %c)
  ret <4 x i32> %res
}

; VMALOF.
define <2 x i64> @test_vmalof(<4 x i32> %a, <4 x i32> %b, <2 x i64> %c) {
; CHECK-LABEL: test_vmalof:
; CHECK: vmalof %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vmalof(<4 x i32> %a, <4 x i32> %b,
                                          <2 x i64> %c)
  ret <2 x i64> %res
}

; VMHB.
define <16 x i8> @test_vmhb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vmhb:
; CHECK: vmhb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vmhb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VMHH.
define <8 x i16> @test_vmhh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vmhh:
; CHECK: vmhh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmhh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VMHF.
define <4 x i32> @test_vmhf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vmhf:
; CHECK: vmhf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmhf(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VMLHB.
define <16 x i8> @test_vmlhb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vmlhb:
; CHECK: vmlhb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vmlhb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VMLHH.
define <8 x i16> @test_vmlhh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vmlhh:
; CHECK: vmlhh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmlhh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VMLHF.
define <4 x i32> @test_vmlhf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vmlhf:
; CHECK: vmlhf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmlhf(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VMEB.
define <8 x i16> @test_vmeb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vmeb:
; CHECK: vmeb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmeb(<16 x i8> %a, <16 x i8> %b)
  ret <8 x i16> %res
}

; VMEH.
define <4 x i32> @test_vmeh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vmeh:
; CHECK: vmeh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmeh(<8 x i16> %a, <8 x i16> %b)
  ret <4 x i32> %res
}

; VMEF.
define <2 x i64> @test_vmef(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vmef:
; CHECK: vmef %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vmef(<4 x i32> %a, <4 x i32> %b)
  ret <2 x i64> %res
}

; VMLEB.
define <8 x i16> @test_vmleb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vmleb:
; CHECK: vmleb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmleb(<16 x i8> %a, <16 x i8> %b)
  ret <8 x i16> %res
}

; VMLEH.
define <4 x i32> @test_vmleh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vmleh:
; CHECK: vmleh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmleh(<8 x i16> %a, <8 x i16> %b)
  ret <4 x i32> %res
}

; VMLEF.
define <2 x i64> @test_vmlef(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vmlef:
; CHECK: vmlef %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vmlef(<4 x i32> %a, <4 x i32> %b)
  ret <2 x i64> %res
}

; VMOB.
define <8 x i16> @test_vmob(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vmob:
; CHECK: vmob %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmob(<16 x i8> %a, <16 x i8> %b)
  ret <8 x i16> %res
}

; VMOH.
define <4 x i32> @test_vmoh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vmoh:
; CHECK: vmoh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmoh(<8 x i16> %a, <8 x i16> %b)
  ret <4 x i32> %res
}

; VMOF.
define <2 x i64> @test_vmof(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vmof:
; CHECK: vmof %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vmof(<4 x i32> %a, <4 x i32> %b)
  ret <2 x i64> %res
}

; VMLOB.
define <8 x i16> @test_vmlob(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vmlob:
; CHECK: vmlob %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vmlob(<16 x i8> %a, <16 x i8> %b)
  ret <8 x i16> %res
}

; VMLOH.
define <4 x i32> @test_vmloh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vmloh:
; CHECK: vmloh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vmloh(<8 x i16> %a, <8 x i16> %b)
  ret <4 x i32> %res
}

; VMLOF.
define <2 x i64> @test_vmlof(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vmlof:
; CHECK: vmlof %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vmlof(<4 x i32> %a, <4 x i32> %b)
  ret <2 x i64> %res
}

; VERLLVB.
define <16 x i8> @test_verllvb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_verllvb:
; CHECK: verllvb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.verllvb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VERLLVH.
define <8 x i16> @test_verllvh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_verllvh:
; CHECK: verllvh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.verllvh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VERLLVF.
define <4 x i32> @test_verllvf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_verllvf:
; CHECK: verllvf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.verllvf(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VERLLVG.
define <2 x i64> @test_verllvg(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_verllvg:
; CHECK: verllvg %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.verllvg(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %res
}

; VERLLB.
define <16 x i8> @test_verllb(<16 x i8> %a, i32 %b) {
; CHECK-LABEL: test_verllb:
; CHECK: verllb %v24, %v24, 0(%r2)
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.verllb(<16 x i8> %a, i32 %b)
  ret <16 x i8> %res
}

; VERLLH.
define <8 x i16> @test_verllh(<8 x i16> %a, i32 %b) {
; CHECK-LABEL: test_verllh:
; CHECK: verllh %v24, %v24, 0(%r2)
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.verllh(<8 x i16> %a, i32 %b)
  ret <8 x i16> %res
}

; VERLLF.
define <4 x i32> @test_verllf(<4 x i32> %a, i32 %b) {
; CHECK-LABEL: test_verllf:
; CHECK: verllf %v24, %v24, 0(%r2)
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.verllf(<4 x i32> %a, i32 %b)
  ret <4 x i32> %res
}

; VERLLG.
define <2 x i64> @test_verllg(<2 x i64> %a, i32 %b) {
; CHECK-LABEL: test_verllg:
; CHECK: verllg %v24, %v24, 0(%r2)
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.verllg(<2 x i64> %a, i32 %b)
  ret <2 x i64> %res
}

; VERLLB with the smallest count.
define <16 x i8> @test_verllb_1(<16 x i8> %a) {
; CHECK-LABEL: test_verllb_1:
; CHECK: verllb %v24, %v24, 1
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.verllb(<16 x i8> %a, i32 1)
  ret <16 x i8> %res
}

; VERLLB with the largest count.
define <16 x i8> @test_verllb_4095(<16 x i8> %a) {
; CHECK-LABEL: test_verllb_4095:
; CHECK: verllb %v24, %v24, 4095
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.verllb(<16 x i8> %a, i32 4095)
  ret <16 x i8> %res
}

; VERLLB with the largest count + 1.
define <16 x i8> @test_verllb_4096(<16 x i8> %a) {
; CHECK-LABEL: test_verllb_4096:
; CHECK: lhi [[REG:%r[1-5]]], 4096
; CHECK: verllb %v24, %v24, 0([[REG]])
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.verllb(<16 x i8> %a, i32 4096)
  ret <16 x i8> %res
}

; VERIMB.
define <16 x i8> @test_verimb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_verimb:
; CHECK: verimb %v24, %v26, %v28, 1
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.verimb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, i32 1)
  ret <16 x i8> %res
}

; VERIMH.
define <8 x i16> @test_verimh(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_verimh:
; CHECK: verimh %v24, %v26, %v28, 1
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.verimh(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c, i32 1)
  ret <8 x i16> %res
}

; VERIMF.
define <4 x i32> @test_verimf(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_verimf:
; CHECK: verimf %v24, %v26, %v28, 1
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.verimf(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c, i32 1)
  ret <4 x i32> %res
}

; VERIMG.
define <2 x i64> @test_verimg(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) {
; CHECK-LABEL: test_verimg:
; CHECK: verimg %v24, %v26, %v28, 1
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.verimg(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c, i32 1)
  ret <2 x i64> %res
}

; VERIMB with a different mask.
define <16 x i8> @test_verimb_254(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_verimb_254:
; CHECK: verimb %v24, %v26, %v28, 254
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.verimb(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c, i32 254)
  ret <16 x i8> %res
}

; VSL.
define <16 x i8> @test_vsl(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsl:
; CHECK: vsl %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsl(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VSLB.
define <16 x i8> @test_vslb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vslb:
; CHECK: vslb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vslb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VSRA.
define <16 x i8> @test_vsra(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsra:
; CHECK: vsra %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsra(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VSRAB.
define <16 x i8> @test_vsrab(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsrab:
; CHECK: vsrab %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsrab(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VSRL.
define <16 x i8> @test_vsrl(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsrl:
; CHECK: vsrl %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsrl(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VSRLB.
define <16 x i8> @test_vsrlb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsrlb:
; CHECK: vsrlb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsrlb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VSLDB with the minimum useful value.
define <16 x i8> @test_vsldb_1(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsldb_1:
; CHECK: vsldb %v24, %v24, %v26, 1
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsldb(<16 x i8> %a, <16 x i8> %b, i32 1)
  ret <16 x i8> %res
}

; VSLDB with the maximum value.
define <16 x i8> @test_vsldb_15(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsldb_15:
; CHECK: vsldb %v24, %v24, %v26, 15
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsldb(<16 x i8> %a, <16 x i8> %b, i32 15)
  ret <16 x i8> %res
}

; VSCBIB.
define <16 x i8> @test_vscbib(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vscbib:
; CHECK: vscbib %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vscbib(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VSCBIH.
define <8 x i16> @test_vscbih(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vscbih:
; CHECK: vscbih %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vscbih(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VSCBIF.
define <4 x i32> @test_vscbif(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vscbif:
; CHECK: vscbif %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vscbif(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VSCBIG.
define <2 x i64> @test_vscbig(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vscbig:
; CHECK: vscbig %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vscbig(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %res
}

; VSQ.
define <16 x i8> @test_vsq(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsq:
; CHECK: vsq %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsq(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VSBIQ.
define <16 x i8> @test_vsbiq(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vsbiq:
; CHECK: vsbiq %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsbiq(<16 x i8> %a, <16 x i8> %b,
                                         <16 x i8> %c)
  ret <16 x i8> %res
}

; VSCBIQ.
define <16 x i8> @test_vscbiq(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vscbiq:
; CHECK: vscbiq %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vscbiq(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VSBCBIQ.
define <16 x i8> @test_vsbcbiq(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vsbcbiq:
; CHECK: vsbcbiq %v24, %v24, %v26, %v28
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsbcbiq(<16 x i8> %a, <16 x i8> %b,
                                           <16 x i8> %c)
  ret <16 x i8> %res
}

; VSUMB.
define <4 x i32> @test_vsumb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsumb:
; CHECK: vsumb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vsumb(<16 x i8> %a, <16 x i8> %b)
  ret <4 x i32> %res
}

; VSUMH.
define <4 x i32> @test_vsumh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsumh:
; CHECK: vsumh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vsumh(<8 x i16> %a, <8 x i16> %b)
  ret <4 x i32> %res
}

; VSUMGH.
define <2 x i64> @test_vsumgh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsumgh:
; CHECK: vsumgh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vsumgh(<8 x i16> %a, <8 x i16> %b)
  ret <2 x i64> %res
}

; VSUMGF.
define <2 x i64> @test_vsumgf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsumgf:
; CHECK: vsumgf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <2 x i64> @llvm.s390.vsumgf(<4 x i32> %a, <4 x i32> %b)
  ret <2 x i64> %res
}

; VSUMQF.
define <16 x i8> @test_vsumqf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsumqf:
; CHECK: vsumqf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsumqf(<4 x i32> %a, <4 x i32> %b)
  ret <16 x i8> %res
}

; VSUMQG.
define <16 x i8> @test_vsumqg(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vsumqg:
; CHECK: vsumqg %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vsumqg(<2 x i64> %a, <2 x i64> %b)
  ret <16 x i8> %res
}

; VTM with no processing of the result.
define i32 @test_vtm(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vtm:
; CHECK: vtm %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %res = call i32 @llvm.s390.vtm(<16 x i8> %a, <16 x i8> %b)
  ret i32 %res
}

; VTM, storing to %ptr if all bits are set.
define void @test_vtm_all_store(<16 x i8> %a, <16 x i8> %b, i32 *%ptr) {
; CHECK-LABEL: test_vtm_all_store:
; CHECK-NOT: %r
; CHECK: vtm %v24, %v26
; CHECK-NEXT: {{bnor|bler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %res = call i32 @llvm.s390.vtm(<16 x i8> %a, <16 x i8> %b)
  %cmp = icmp sge i32 %res, 3
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret void
}

; VCEQBS with no processing of the result.
define i32 @test_vceqbs(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vceqbs:
; CHECK: vceqbs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vceqbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 1
  ret i32 %res
}

; VCEQBS, returning 1 if any elements are equal (CC != 3).
define i32 @test_vceqbs_any_bool(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vceqbs_any_bool:
; CHECK: vceqbs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochile %r2, 1
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vceqbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 1
  %cmp = icmp ne i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCEQBS, storing to %ptr if any elements are equal.
define <16 x i8> @test_vceqbs_any_store(<16 x i8> %a, <16 x i8> %b, i32 *%ptr) {
; CHECK-LABEL: test_vceqbs_any_store:
; CHECK-NOT: %r
; CHECK: vceqbs %v24, %v24, %v26
; CHECK-NEXT: {{bor|bnler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vceqbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  %cmp = icmp ule i32 %cc, 2
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <16 x i8> %res
}

; VCEQHS with no processing of the result.
define i32 @test_vceqhs(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vceqhs:
; CHECK: vceqhs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vceqhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 1
  ret i32 %res
}

; VCEQHS, returning 1 if not all elements are equal.
define i32 @test_vceqhs_notall_bool(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vceqhs_notall_bool:
; CHECK: vceqhs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochinhe %r2, 1
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vceqhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 1
  %cmp = icmp sge i32 %res, 1
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCEQHS, storing to %ptr if not all elements are equal.
define <8 x i16> @test_vceqhs_notall_store(<8 x i16> %a, <8 x i16> %b,
                                           i32 *%ptr) {
; CHECK-LABEL: test_vceqhs_notall_store:
; CHECK-NOT: %r
; CHECK: vceqhs %v24, %v24, %v26
; CHECK-NEXT: {{bher|ber}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vceqhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  %cmp = icmp ugt i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <8 x i16> %res
}

; VCEQFS with no processing of the result.
define i32 @test_vceqfs(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vceqfs:
; CHECK: vceqfs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vceqfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  ret i32 %res
}

; VCEQFS, returning 1 if no elements are equal.
define i32 @test_vceqfs_none_bool(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vceqfs_none_bool:
; CHECK: vceqfs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochio %r2, 1
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vceqfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp eq i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCEQFS, storing to %ptr if no elements are equal.
define <4 x i32> @test_vceqfs_none_store(<4 x i32> %a, <4 x i32> %b,
                                         i32 *%ptr) {
; CHECK-LABEL: test_vceqfs_none_store:
; CHECK-NOT: %r
; CHECK: vceqfs %v24, %v24, %v26
; CHECK-NEXT: {{bnor|bler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vceqfs(<4 x i32> %a, <4 x i32> %b)
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

; VCEQGS with no processing of the result.
define i32 @test_vceqgs(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vceqgs:
; CHECK: vceqgs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vceqgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  ret i32 %res
}

; VCEQGS returning 1 if all elements are equal (CC == 0).
define i32 @test_vceqgs_all_bool(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vceqgs_all_bool:
; CHECK: vceqgs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochie %r2, 1
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vceqgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp ult i32 %res, 1
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCEQGS, storing to %ptr if all elements are equal.
define <2 x i64> @test_vceqgs_all_store(<2 x i64> %a, <2 x i64> %b, i32 *%ptr) {
; CHECK-LABEL: test_vceqgs_all_store:
; CHECK-NOT: %r
; CHECK: vceqgs %v24, %v24, %v26
; CHECK-NEXT: {{bnher|bner}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vceqgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 0
  %cc = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp sle i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <2 x i64> %res
}

; VCHBS with no processing of the result.
define i32 @test_vchbs(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vchbs:
; CHECK: vchbs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vchbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 1
  ret i32 %res
}

; VCHBS, returning 1 if any elements are higher (CC != 3).
define i32 @test_vchbs_any_bool(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vchbs_any_bool:
; CHECK: vchbs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochile %r2, 1
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vchbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 1
  %cmp = icmp ne i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCHBS, storing to %ptr if any elements are higher.
define <16 x i8> @test_vchbs_any_store(<16 x i8> %a, <16 x i8> %b, i32 *%ptr) {
; CHECK-LABEL: test_vchbs_any_store:
; CHECK-NOT: %r
; CHECK: vchbs %v24, %v24, %v26
; CHECK-NEXT: {{bor|bnler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vchbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  %cmp = icmp ule i32 %cc, 2
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <16 x i8> %res
}

; VCHHS with no processing of the result.
define i32 @test_vchhs(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vchhs:
; CHECK: vchhs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vchhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 1
  ret i32 %res
}

; VCHHS, returning 1 if not all elements are higher.
define i32 @test_vchhs_notall_bool(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vchhs_notall_bool:
; CHECK: vchhs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochinhe %r2, 1
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vchhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 1
  %cmp = icmp sge i32 %res, 1
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCHHS, storing to %ptr if not all elements are higher.
define <8 x i16> @test_vchhs_notall_store(<8 x i16> %a, <8 x i16> %b,
                                          i32 *%ptr) {
; CHECK-LABEL: test_vchhs_notall_store:
; CHECK-NOT: %r
; CHECK: vchhs %v24, %v24, %v26
; CHECK-NEXT: {{bher|ber}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vchhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  %cmp = icmp ugt i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <8 x i16> %res
}

; VCHFS with no processing of the result.
define i32 @test_vchfs(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vchfs:
; CHECK: vchfs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vchfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  ret i32 %res
}

; VCHFS, returning 1 if no elements are higher.
define i32 @test_vchfs_none_bool(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vchfs_none_bool:
; CHECK: vchfs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochio %r2, 1
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vchfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp eq i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCHFS, storing to %ptr if no elements are higher.
define <4 x i32> @test_vchfs_none_store(<4 x i32> %a, <4 x i32> %b, i32 *%ptr) {
; CHECK-LABEL: test_vchfs_none_store:
; CHECK-NOT: %r
; CHECK: vchfs %v24, %v24, %v26
; CHECK-NEXT: {{bnor|bler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vchfs(<4 x i32> %a, <4 x i32> %b)
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

; VCHGS with no processing of the result.
define i32 @test_vchgs(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vchgs:
; CHECK: vchgs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vchgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  ret i32 %res
}

; VCHGS returning 1 if all elements are higher (CC == 0).
define i32 @test_vchgs_all_bool(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vchgs_all_bool:
; CHECK: vchgs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochie %r2, 1
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vchgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp ult i32 %res, 1
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCHGS, storing to %ptr if all elements are higher.
define <2 x i64> @test_vchgs_all_store(<2 x i64> %a, <2 x i64> %b, i32 *%ptr) {
; CHECK-LABEL: test_vchgs_all_store:
; CHECK-NOT: %r
; CHECK: vchgs %v24, %v24, %v26
; CHECK-NEXT: {{bnher|bner}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vchgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 0
  %cc = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp sle i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <2 x i64> %res
}

; VCHLBS with no processing of the result.
define i32 @test_vchlbs(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vchlbs:
; CHECK: vchlbs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vchlbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 1
  ret i32 %res
}

; VCHLBS, returning 1 if any elements are higher (CC != 3).
define i32 @test_vchlbs_any_bool(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vchlbs_any_bool:
; CHECK: vchlbs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochile %r2, 1
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vchlbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 1
  %cmp = icmp ne i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCHLBS, storing to %ptr if any elements are higher.
define <16 x i8> @test_vchlbs_any_store(<16 x i8> %a, <16 x i8> %b, i32 *%ptr) {
; CHECK-LABEL: test_vchlbs_any_store:
; CHECK-NOT: %r
; CHECK: vchlbs %v24, %v24, %v26
; CHECK-NEXT: {{bor|bnler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vchlbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  %cmp = icmp sle i32 %cc, 2
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <16 x i8> %res
}

; VCHLHS with no processing of the result.
define i32 @test_vchlhs(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vchlhs:
; CHECK: vchlhs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vchlhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 1
  ret i32 %res
}

; VCHLHS, returning 1 if not all elements are higher.
define i32 @test_vchlhs_notall_bool(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vchlhs_notall_bool:
; CHECK: vchlhs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochinhe %r2, 1
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vchlhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 1
  %cmp = icmp uge i32 %res, 1
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCHLHS, storing to %ptr if not all elements are higher.
define <8 x i16> @test_vchlhs_notall_store(<8 x i16> %a, <8 x i16> %b,
                                           i32 *%ptr) {
; CHECK-LABEL: test_vchlhs_notall_store:
; CHECK-NOT: %r
; CHECK: vchlhs %v24, %v24, %v26
; CHECK-NEXT: {{bher|ber}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vchlhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  %cmp = icmp sgt i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <8 x i16> %res
}

; VCHLFS with no processing of the result.
define i32 @test_vchlfs(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vchlfs:
; CHECK: vchlfs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vchlfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  ret i32 %res
}

; VCHLFS, returning 1 if no elements are higher.
define i32 @test_vchlfs_none_bool(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vchlfs_none_bool:
; CHECK: vchlfs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochio %r2, 1
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vchlfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp eq i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCHLFS, storing to %ptr if no elements are higher.
define <4 x i32> @test_vchlfs_none_store(<4 x i32> %a, <4 x i32> %b,
                                         i32 *%ptr) {
; CHECK-LABEL: test_vchlfs_none_store:
; CHECK-NOT: %r
; CHECK: vchlfs %v24, %v24, %v26
; CHECK-NEXT: {{bnor|bler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vchlfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  %cmp = icmp sge i32 %cc, 3
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <4 x i32> %res
}

; VCHLGS with no processing of the result.
define i32 @test_vchlgs(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vchlgs:
; CHECK: vchlgs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vchlgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  ret i32 %res
}

; VCHLGS returning 1 if all elements are higher (CC == 0).
define i32 @test_vchlgs_all_bool(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vchlgs_all_bool:
; CHECK: vchlgs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochie %r2, 1
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vchlgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp slt i32 %res, 1
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VCHLGS, storing to %ptr if all elements are higher.
define <2 x i64> @test_vchlgs_all_store(<2 x i64> %a, <2 x i64> %b, i32 *%ptr) {
; CHECK-LABEL: test_vchlgs_all_store:
; CHECK-NOT: %r
; CHECK: vchlgs %v24, %v24, %v26
; CHECK-NEXT: {{bnher|bner}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vchlgs(<2 x i64> %a, <2 x i64> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 0
  %cc = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp ule i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <2 x i64> %res
}

; VFAEB with !IN !RT.
define <16 x i8> @test_vfaeb_0(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaeb_0:
; CHECK: vfaeb %v24, %v24, %v26, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %a, <16 x i8> %b, i32 0)
  ret <16 x i8> %res
}

; VFAEB with !IN RT.
define <16 x i8> @test_vfaeb_4(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaeb_4:
; CHECK: vfaeb %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %a, <16 x i8> %b, i32 4)
  ret <16 x i8> %res
}

; VFAEB with IN !RT.
define <16 x i8> @test_vfaeb_8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaeb_8:
; CHECK: vfaeb %v24, %v24, %v26, 8
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %a, <16 x i8> %b, i32 8)
  ret <16 x i8> %res
}

; VFAEB with IN RT.
define <16 x i8> @test_vfaeb_12(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaeb_12:
; CHECK: vfaeb %v24, %v24, %v26, 12
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %a, <16 x i8> %b, i32 12)
  ret <16 x i8> %res
}

; VFAEB with CS -- should be ignored.
define <16 x i8> @test_vfaeb_1(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaeb_1:
; CHECK: vfaeb %v24, %v24, %v26, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaeb(<16 x i8> %a, <16 x i8> %b, i32 1)
  ret <16 x i8> %res
}

; VFAEH.
define <8 x i16> @test_vfaeh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vfaeh:
; CHECK: vfaeh %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vfaeh(<8 x i16> %a, <8 x i16> %b, i32 4)
  ret <8 x i16> %res
}

; VFAEF.
define <4 x i32> @test_vfaef(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vfaef:
; CHECK: vfaef %v24, %v24, %v26, 8
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vfaef(<4 x i32> %a, <4 x i32> %b, i32 8)
  ret <4 x i32> %res
}

; VFAEBS.
define <16 x i8> @test_vfaebs(<16 x i8> %a, <16 x i8> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfaebs:
; CHECK: vfaebs %v24, %v24, %v26, 0
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vfaebs(<16 x i8> %a, <16 x i8> %b,
                                                  i32 0)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VFAEHS.
define <8 x i16> @test_vfaehs(<8 x i16> %a, <8 x i16> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfaehs:
; CHECK: vfaehs %v24, %v24, %v26, 4
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vfaehs(<8 x i16> %a, <8 x i16> %b,
                                                  i32 4)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VFAEFS.
define <4 x i32> @test_vfaefs(<4 x i32> %a, <4 x i32> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfaefs:
; CHECK: vfaefs %v24, %v24, %v26, 8
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfaefs(<4 x i32> %a, <4 x i32> %b,
                                                  i32 8)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VFAEZB with !IN !RT.
define <16 x i8> @test_vfaezb_0(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaezb_0:
; CHECK: vfaezb %v24, %v24, %v26, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %a, <16 x i8> %b, i32 0)
  ret <16 x i8> %res
}

; VFAEZB with !IN RT.
define <16 x i8> @test_vfaezb_4(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaezb_4:
; CHECK: vfaezb %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %a, <16 x i8> %b, i32 4)
  ret <16 x i8> %res
}

; VFAEZB with IN !RT.
define <16 x i8> @test_vfaezb_8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaezb_8:
; CHECK: vfaezb %v24, %v24, %v26, 8
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %a, <16 x i8> %b, i32 8)
  ret <16 x i8> %res
}

; VFAEZB with IN RT.
define <16 x i8> @test_vfaezb_12(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaezb_12:
; CHECK: vfaezb %v24, %v24, %v26, 12
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %a, <16 x i8> %b, i32 12)
  ret <16 x i8> %res
}

; VFAEZB with CS -- should be ignored.
define <16 x i8> @test_vfaezb_1(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfaezb_1:
; CHECK: vfaezb %v24, %v24, %v26, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfaezb(<16 x i8> %a, <16 x i8> %b, i32 1)
  ret <16 x i8> %res
}

; VFAEZH.
define <8 x i16> @test_vfaezh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vfaezh:
; CHECK: vfaezh %v24, %v24, %v26, 4
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vfaezh(<8 x i16> %a, <8 x i16> %b, i32 4)
  ret <8 x i16> %res
}

; VFAEZF.
define <4 x i32> @test_vfaezf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vfaezf:
; CHECK: vfaezf %v24, %v24, %v26, 8
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vfaezf(<4 x i32> %a, <4 x i32> %b, i32 8)
  ret <4 x i32> %res
}

; VFAEZBS.
define <16 x i8> @test_vfaezbs(<16 x i8> %a, <16 x i8> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfaezbs:
; CHECK: vfaezbs %v24, %v24, %v26, 0
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vfaezbs(<16 x i8> %a, <16 x i8> %b,
                                                   i32 0)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VFAEZHS.
define <8 x i16> @test_vfaezhs(<8 x i16> %a, <8 x i16> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfaezhs:
; CHECK: vfaezhs %v24, %v24, %v26, 4
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vfaezhs(<8 x i16> %a, <8 x i16> %b,
                                                   i32 4)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VFAEZFS.
define <4 x i32> @test_vfaezfs(<4 x i32> %a, <4 x i32> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfaezfs:
; CHECK: vfaezfs %v24, %v24, %v26, 8
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfaezfs(<4 x i32> %a, <4 x i32> %b,
                                                   i32 8)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VFEEB.
define <16 x i8> @test_vfeeb_0(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfeeb_0:
; CHECK: vfeeb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfeeb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VFEEH.
define <8 x i16> @test_vfeeh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vfeeh:
; CHECK: vfeeh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vfeeh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VFEEF.
define <4 x i32> @test_vfeef(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vfeef:
; CHECK: vfeef %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vfeef(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VFEEBS.
define <16 x i8> @test_vfeebs(<16 x i8> %a, <16 x i8> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfeebs:
; CHECK: vfeebs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vfeebs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VFEEHS.
define <8 x i16> @test_vfeehs(<8 x i16> %a, <8 x i16> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfeehs:
; CHECK: vfeehs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vfeehs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VFEEFS.
define <4 x i32> @test_vfeefs(<4 x i32> %a, <4 x i32> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfeefs:
; CHECK: vfeefs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfeefs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VFEEZB.
define <16 x i8> @test_vfeezb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfeezb:
; CHECK: vfeezb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfeezb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VFEEZH.
define <8 x i16> @test_vfeezh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vfeezh:
; CHECK: vfeezh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vfeezh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VFEEZF.
define <4 x i32> @test_vfeezf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vfeezf:
; CHECK: vfeezf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vfeezf(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VFEEZBS.
define <16 x i8> @test_vfeezbs(<16 x i8> %a, <16 x i8> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfeezbs:
; CHECK: vfeezbs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vfeezbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VFEEZHS.
define <8 x i16> @test_vfeezhs(<8 x i16> %a, <8 x i16> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfeezhs:
; CHECK: vfeezhs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vfeezhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VFEEZFS.
define <4 x i32> @test_vfeezfs(<4 x i32> %a, <4 x i32> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfeezfs:
; CHECK: vfeezfs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfeezfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VFENEB.
define <16 x i8> @test_vfeneb_0(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfeneb_0:
; CHECK: vfeneb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfeneb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VFENEH.
define <8 x i16> @test_vfeneh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vfeneh:
; CHECK: vfeneh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vfeneh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VFENEF.
define <4 x i32> @test_vfenef(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vfenef:
; CHECK: vfenef %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vfenef(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VFENEBS.
define <16 x i8> @test_vfenebs(<16 x i8> %a, <16 x i8> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfenebs:
; CHECK: vfenebs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vfenebs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VFENEHS.
define <8 x i16> @test_vfenehs(<8 x i16> %a, <8 x i16> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfenehs:
; CHECK: vfenehs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vfenehs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VFENEFS.
define <4 x i32> @test_vfenefs(<4 x i32> %a, <4 x i32> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfenefs:
; CHECK: vfenefs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfenefs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VFENEZB.
define <16 x i8> @test_vfenezb(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vfenezb:
; CHECK: vfenezb %v24, %v24, %v26
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vfenezb(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %res
}

; VFENEZH.
define <8 x i16> @test_vfenezh(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vfenezh:
; CHECK: vfenezh %v24, %v24, %v26
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vfenezh(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %res
}

; VFENEZF.
define <4 x i32> @test_vfenezf(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vfenezf:
; CHECK: vfenezf %v24, %v24, %v26
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vfenezf(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %res
}

; VFENEZBS.
define <16 x i8> @test_vfenezbs(<16 x i8> %a, <16 x i8> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfenezbs:
; CHECK: vfenezbs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vfenezbs(<16 x i8> %a, <16 x i8> %b)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VFENEZHS.
define <8 x i16> @test_vfenezhs(<8 x i16> %a, <8 x i16> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfenezhs:
; CHECK: vfenezhs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vfenezhs(<8 x i16> %a, <8 x i16> %b)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VFENEZFS.
define <4 x i32> @test_vfenezfs(<4 x i32> %a, <4 x i32> %b, i32 *%ccptr) {
; CHECK-LABEL: test_vfenezfs:
; CHECK: vfenezfs %v24, %v24, %v26
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vfenezfs(<4 x i32> %a, <4 x i32> %b)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VISTRB.
define <16 x i8> @test_vistrb(<16 x i8> %a) {
; CHECK-LABEL: test_vistrb:
; CHECK: vistrb %v24, %v24
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vistrb(<16 x i8> %a)
  ret <16 x i8> %res
}

; VISTRH.
define <8 x i16> @test_vistrh(<8 x i16> %a) {
; CHECK-LABEL: test_vistrh:
; CHECK: vistrh %v24, %v24
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vistrh(<8 x i16> %a)
  ret <8 x i16> %res
}

; VISTRF.
define <4 x i32> @test_vistrf(<4 x i32> %a) {
; CHECK-LABEL: test_vistrf:
; CHECK: vistrf %v24, %v24
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vistrf(<4 x i32> %a)
  ret <4 x i32> %res
}

; VISTRBS.
define <16 x i8> @test_vistrbs(<16 x i8> %a, i32 *%ccptr) {
; CHECK-LABEL: test_vistrbs:
; CHECK: vistrbs %v24, %v24
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vistrbs(<16 x i8> %a)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VISTRHS.
define <8 x i16> @test_vistrhs(<8 x i16> %a, i32 *%ccptr) {
; CHECK-LABEL: test_vistrhs:
; CHECK: vistrhs %v24, %v24
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vistrhs(<8 x i16> %a)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VISTRFS.
define <4 x i32> @test_vistrfs(<4 x i32> %a, i32 *%ccptr) {
; CHECK-LABEL: test_vistrfs:
; CHECK: vistrfs %v24, %v24
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vistrfs(<4 x i32> %a)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VSTRCB with !IN !RT.
define <16 x i8> @test_vstrcb_0(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrcb_0:
; CHECK: vstrcb %v24, %v24, %v26, %v28, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %a, <16 x i8> %b,
                                          <16 x i8> %c, i32 0)
  ret <16 x i8> %res
}

; VSTRCB with !IN RT.
define <16 x i8> @test_vstrcb_4(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrcb_4:
; CHECK: vstrcb %v24, %v24, %v26, %v28, 4
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %a, <16 x i8> %b,
                                          <16 x i8> %c, i32 4)
  ret <16 x i8> %res
}

; VSTRCB with IN !RT.
define <16 x i8> @test_vstrcb_8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrcb_8:
; CHECK: vstrcb %v24, %v24, %v26, %v28, 8
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %a, <16 x i8> %b,
                                          <16 x i8> %c, i32 8)
  ret <16 x i8> %res
}

; VSTRCB with IN RT.
define <16 x i8> @test_vstrcb_12(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrcb_12:
; CHECK: vstrcb %v24, %v24, %v26, %v28, 12
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %a, <16 x i8> %b,
                                          <16 x i8> %c, i32 12)
  ret <16 x i8> %res
}

; VSTRCB with CS -- should be ignored.
define <16 x i8> @test_vstrcb_1(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrcb_1:
; CHECK: vstrcb %v24, %v24, %v26, %v28, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrcb(<16 x i8> %a, <16 x i8> %b,
                                          <16 x i8> %c, i32 1)
  ret <16 x i8> %res
}

; VSTRCH.
define <8 x i16> @test_vstrch(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vstrch:
; CHECK: vstrch %v24, %v24, %v26, %v28, 4
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vstrch(<8 x i16> %a, <8 x i16> %b,
                                          <8 x i16> %c, i32 4)
  ret <8 x i16> %res
}

; VSTRCF.
define <4 x i32> @test_vstrcf(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vstrcf:
; CHECK: vstrcf %v24, %v24, %v26, %v28, 8
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vstrcf(<4 x i32> %a, <4 x i32> %b,
                                          <4 x i32> %c, i32 8)
  ret <4 x i32> %res
}

; VSTRCBS.
define <16 x i8> @test_vstrcbs(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c,
                               i32 *%ccptr) {
; CHECK-LABEL: test_vstrcbs:
; CHECK: vstrcbs %v24, %v24, %v26, %v28, 0
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vstrcbs(<16 x i8> %a, <16 x i8> %b,
                                                   <16 x i8> %c, i32 0)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VSTRCHS.
define <8 x i16> @test_vstrchs(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c,
                               i32 *%ccptr) {
; CHECK-LABEL: test_vstrchs:
; CHECK: vstrchs %v24, %v24, %v26, %v28, 4
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vstrchs(<8 x i16> %a, <8 x i16> %b,
                                                   <8 x i16> %c, i32 4)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VSTRCFS.
define <4 x i32> @test_vstrcfs(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c,
                               i32 *%ccptr) {
; CHECK-LABEL: test_vstrcfs:
; CHECK: vstrcfs %v24, %v24, %v26, %v28, 8
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vstrcfs(<4 x i32> %a, <4 x i32> %b,
                                                   <4 x i32> %c, i32 8)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VSTRCZB with !IN !RT.
define <16 x i8> @test_vstrczb_0(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrczb_0:
; CHECK: vstrczb %v24, %v24, %v26, %v28, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrczb(<16 x i8> %a, <16 x i8> %b,
                                           <16 x i8> %c, i32 0)
  ret <16 x i8> %res
}

; VSTRCZB with !IN RT.
define <16 x i8> @test_vstrczb_4(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrczb_4:
; CHECK: vstrczb %v24, %v24, %v26, %v28, 4
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrczb(<16 x i8> %a, <16 x i8> %b,
                                           <16 x i8> %c, i32 4)
  ret <16 x i8> %res
}

; VSTRCZB with IN !RT.
define <16 x i8> @test_vstrczb_8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrczb_8:
; CHECK: vstrczb %v24, %v24, %v26, %v28, 8
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrczb(<16 x i8> %a, <16 x i8> %b,
                                           <16 x i8> %c, i32 8)
  ret <16 x i8> %res
}

; VSTRCZB with IN RT.
define <16 x i8> @test_vstrczb_12(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrczb_12:
; CHECK: vstrczb %v24, %v24, %v26, %v28, 12
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrczb(<16 x i8> %a, <16 x i8> %b,
                                           <16 x i8> %c, i32 12)
  ret <16 x i8> %res
}

; VSTRCZB with CS -- should be ignored.
define <16 x i8> @test_vstrczb_1(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vstrczb_1:
; CHECK: vstrczb %v24, %v24, %v26, %v28, 0
; CHECK: br %r14
  %res = call <16 x i8> @llvm.s390.vstrczb(<16 x i8> %a, <16 x i8> %b,
                                           <16 x i8> %c, i32 1)
  ret <16 x i8> %res
}

; VSTRCZH.
define <8 x i16> @test_vstrczh(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vstrczh:
; CHECK: vstrczh %v24, %v24, %v26, %v28, 4
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vstrczh(<8 x i16> %a, <8 x i16> %b,
                                           <8 x i16> %c,  i32 4)
  ret <8 x i16> %res
}

; VSTRCZF.
define <4 x i32> @test_vstrczf(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vstrczf:
; CHECK: vstrczf %v24, %v24, %v26, %v28, 8
; CHECK: br %r14
  %res = call <4 x i32> @llvm.s390.vstrczf(<4 x i32> %a, <4 x i32> %b,
                                           <4 x i32> %c, i32 8)
  ret <4 x i32> %res
}

; VSTRCZBS.
define <16 x i8> @test_vstrczbs(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c,
                                i32 *%ccptr) {
; CHECK-LABEL: test_vstrczbs:
; CHECK: vstrczbs %v24, %v24, %v26, %v28, 0
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<16 x i8>, i32} @llvm.s390.vstrczbs(<16 x i8> %a, <16 x i8> %b,
                                                    <16 x i8> %c, i32 0)
  %res = extractvalue {<16 x i8>, i32} %call, 0
  %cc = extractvalue {<16 x i8>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <16 x i8> %res
}

; VSTRCZHS.
define <8 x i16> @test_vstrczhs(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c,
                                i32 *%ccptr) {
; CHECK-LABEL: test_vstrczhs:
; CHECK: vstrczhs %v24, %v24, %v26, %v28, 4
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<8 x i16>, i32} @llvm.s390.vstrczhs(<8 x i16> %a, <8 x i16> %b,
                                                    <8 x i16> %c, i32 4)
  %res = extractvalue {<8 x i16>, i32} %call, 0
  %cc = extractvalue {<8 x i16>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <8 x i16> %res
}

; VSTRCZFS.
define <4 x i32> @test_vstrczfs(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c,
                                i32 *%ccptr) {
; CHECK-LABEL: test_vstrczfs:
; CHECK: vstrczfs %v24, %v24, %v26, %v28, 8
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: srl [[REG]], 28
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %call = call {<4 x i32>, i32} @llvm.s390.vstrczfs(<4 x i32> %a, <4 x i32> %b,
                                                    <4 x i32> %c, i32 8)
  %res = extractvalue {<4 x i32>, i32} %call, 0
  %cc = extractvalue {<4 x i32>, i32} %call, 1
  store i32 %cc, i32 *%ccptr
  ret <4 x i32> %res
}

; VFCEDBS with no processing of the result.
define i32 @test_vfcedbs(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfcedbs:
; CHECK: vfcedbs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vfcedbs(<2 x double> %a,
                                                   <2 x double> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  ret i32 %res
}

; VFCEDBS, returning 1 if any elements are equal (CC != 3).
define i32 @test_vfcedbs_any_bool(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfcedbs_any_bool:
; CHECK: vfcedbs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochile %r2, 1
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vfcedbs(<2 x double> %a,
                                                   <2 x double> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp ne i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VFCEDBS, storing to %ptr if any elements are equal.
define <2 x i64> @test_vfcedbs_any_store(<2 x double> %a, <2 x double> %b,
                                         i32 *%ptr) {
; CHECK-LABEL: test_vfcedbs_any_store:
; CHECK-NOT: %r
; CHECK: vfcedbs %v24, %v24, %v26
; CHECK-NEXT: {{bor|bnler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vfcedbs(<2 x double> %a,
                                                   <2 x double> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 0
  %cc = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp ule i32 %cc, 2
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <2 x i64> %res
}

; VFCHDBS with no processing of the result.
define i32 @test_vfchdbs(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfchdbs:
; CHECK: vfchdbs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vfchdbs(<2 x double> %a,
                                                   <2 x double> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  ret i32 %res
}

; VFCHDBS, returning 1 if not all elements are higher.
define i32 @test_vfchdbs_notall_bool(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfchdbs_notall_bool:
; CHECK: vfchdbs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochinhe %r2, 1
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vfchdbs(<2 x double> %a,
                                                   <2 x double> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp sge i32 %res, 1
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VFCHDBS, storing to %ptr if not all elements are higher.
define <2 x i64> @test_vfchdbs_notall_store(<2 x double> %a, <2 x double> %b,
                                            i32 *%ptr) {
; CHECK-LABEL: test_vfchdbs_notall_store:
; CHECK-NOT: %r
; CHECK: vfchdbs %v24, %v24, %v26
; CHECK-NEXT: {{bher|ber}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vfchdbs(<2 x double> %a,
                                                   <2 x double> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 0
  %cc = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp ugt i32 %cc, 0
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <2 x i64> %res
}

; VFCHEDBS with no processing of the result.
define i32 @test_vfchedbs(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfchedbs:
; CHECK: vfchedbs {{%v[0-9]+}}, %v24, %v26
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vfchedbs(<2 x double> %a,
						    <2 x double> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  ret i32 %res
}

; VFCHEDBS, returning 1 if neither element is higher or equal.
define i32 @test_vfchedbs_none_bool(<2 x double> %a, <2 x double> %b) {
; CHECK-LABEL: test_vfchedbs_none_bool:
; CHECK: vfchedbs {{%v[0-9]+}}, %v24, %v26
; CHECK: lhi %r2, 0
; CHECK: lochio %r2, 1
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vfchedbs(<2 x double> %a,
						    <2 x double> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp eq i32 %res, 3
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VFCHEDBS, storing to %ptr if neither element is higher or equal.
define <2 x i64> @test_vfchedbs_none_store(<2 x double> %a, <2 x double> %b,
                                           i32 *%ptr) {
; CHECK-LABEL: test_vfchedbs_none_store:
; CHECK-NOT: %r
; CHECK: vfchedbs %v24, %v24, %v26
; CHECK-NEXT: {{bnor|bler}} %r14
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vfchedbs(<2 x double> %a,
						    <2 x double> %b)
  %res = extractvalue {<2 x i64>, i32} %call, 0
  %cc = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp uge i32 %cc, 3
  br i1 %cmp, label %store, label %exit

store:
  store i32 0, i32 *%ptr
  br label %exit

exit:
  ret <2 x i64> %res
}

; VFTCIDB with the lowest useful class selector and no processing of the result.
define i32 @test_vftcidb(<2 x double> %a) {
; CHECK-LABEL: test_vftcidb:
; CHECK: vftcidb {{%v[0-9]+}}, %v24, 1
; CHECK: ipm %r2
; CHECK: srl %r2, 28
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vftcidb(<2 x double> %a, i32 1)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  ret i32 %res
}

; VFTCIDB with the highest useful class selector, returning 1 if all elements
; have the right class (CC == 0).
define i32 @test_vftcidb_all_bool(<2 x double> %a) {
; CHECK-LABEL: test_vftcidb_all_bool:
; CHECK: vftcidb {{%v[0-9]+}}, %v24, 4094
; CHECK: lhi %r2, 0
; CHECK: lochie %r2, 1
; CHECK: br %r14
  %call = call {<2 x i64>, i32} @llvm.s390.vftcidb(<2 x double> %a, i32 4094)
  %res = extractvalue {<2 x i64>, i32} %call, 1
  %cmp = icmp eq i32 %res, 0
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}

; VFIDB with a rounding mode not usable via standard intrinsics.
define <2 x double> @test_vfidb_0_4(<2 x double> %a) {
; CHECK-LABEL: test_vfidb_0_4:
; CHECK: vfidb %v24, %v24, 0, 4
; CHECK: br %r14
  %res = call <2 x double> @llvm.s390.vfidb(<2 x double> %a, i32 0, i32 4)
  ret <2 x double> %res
}

; VFIDB with IEEE-inexact exception suppressed.
define <2 x double> @test_vfidb_4_0(<2 x double> %a) {
; CHECK-LABEL: test_vfidb_4_0:
; CHECK: vfidb %v24, %v24, 4, 0
; CHECK: br %r14
  %res = call <2 x double> @llvm.s390.vfidb(<2 x double> %a, i32 4, i32 0)
  ret <2 x double> %res
}

