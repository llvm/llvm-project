; RUN: llc -disable-strictnode-mutation < %s -mtriple=i686-unknown-unknown -mattr=+sse2 -O3 | FileCheck %s --check-prefixes=CHECK,SSE
; RUN: llc -disable-strictnode-mutation < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 -O3 | FileCheck %s --check-prefixes=CHECK,SSE
; RUN: llc -disable-strictnode-mutation < %s -mtriple=i686-unknown-unknown -mattr=+avx -O3 | FileCheck %s --check-prefixes=CHECK,AVX
; RUN: llc -disable-strictnode-mutation < %s -mtriple=x86_64-unknown-unknown -mattr=+avx -O3 | FileCheck %s --check-prefixes=CHECK,AVX
; RUN: llc -disable-strictnode-mutation < %s -mtriple=i686-unknown-unknown -mattr=+avx512f -mattr=+avx512vl -O3 | FileCheck %s --check-prefixes=CHECK,AVX512-32
; RUN: llc -disable-strictnode-mutation < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512f -mattr=+avx512vl -O3 | FileCheck %s --check-prefixes=CHECK,AVX512-64

define <4 x i32> @test_v4f32_oeq_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_oeq_q:
; SSE:       # %bb.0:
; SSE:         cmpeqps {{.*}}, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_oeq_q:
; AVX:       # %bb.0:
; AVX:         vcmpeqps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_oeq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_oeq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeqps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ogt_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ogt_q:
; SSE:       # %bb.0:
; SSE:         ucomiss %xmm4, %xmm5
; SSE:         unpckhpd {{.*#+}} xmm5 = xmm5[1],xmm3[1]
; SSE:         unpckhpd {{.*#+}} xmm6 = xmm6[1],xmm2[1]
; SSE:         ucomiss %xmm5, %xmm6
; SSE:         ucomiss %xmm3, %xmm2
; SSE:         ucomiss %xmm3, %xmm2
;
; AVX-LABEL: test_v4f32_ogt_q:
; AVX:       # %bb.0:
; AVX:         vcmplt_oqps {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ogt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgt_oqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ogt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqps %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_oge_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_oge_q:
; SSE:       # %bb.0:
; SSE:         ucomiss %xmm4, %xmm5
; SSE:         unpckhpd {{.*#+}} xmm5 = xmm5[1],xmm3[1]
; SSE:         unpckhpd {{.*#+}} xmm6 = xmm6[1],xmm2[1]
; SSE:         ucomiss %xmm5, %xmm6
; SSE:         ucomiss %xmm3, %xmm2
; SSE:         ucomiss %xmm3, %xmm2
;
; AVX-LABEL: test_v4f32_oge_q:
; AVX:       # %bb.0:
; AVX:         vcmple_oqps {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_oge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpge_oqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_oge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqps %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_olt_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_olt_q:
; SSE:       # %bb.0:
; SSE:         ucomiss %xmm4, %xmm5
; SSE:         unpckhpd {{.*#+}} xmm5 = xmm5[1],xmm2[1]
; SSE:         unpckhpd {{.*#+}} xmm6 = xmm6[1],xmm3[1]
; SSE:         ucomiss %xmm5, %xmm6
; SSE:         ucomiss %xmm2, %xmm3
; SSE:         ucomiss %xmm2, %xmm3
;
; AVX-LABEL: test_v4f32_olt_q:
; AVX:       # %bb.0:
; AVX:         vcmplt_oqps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_olt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmplt_oqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_olt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ole_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ole_q:
; SSE:       # %bb.0:
; SSE:         ucomiss %xmm4, %xmm5
; SSE:         unpckhpd {{.*#+}} xmm5 = xmm5[1],xmm2[1]
; SSE:         unpckhpd {{.*#+}} xmm6 = xmm6[1],xmm3[1]
; SSE:         ucomiss %xmm5, %xmm6
; SSE:         ucomiss %xmm2, %xmm3
; SSE:         ucomiss %xmm2, %xmm3
;
; AVX-LABEL: test_v4f32_ole_q:
; AVX:       # %bb.0:
; AVX:         vcmple_oqps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ole_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmple_oqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ole_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_one_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_one_q:
; SSE:       # %bb.0:
; SSE:         cmpneqps %xmm3, %xmm4
; SSE-NEXT:    cmpordps %xmm3, %xmm2
; SSE-NEXT:    andps %xmm4, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_one_q:
; AVX:       # %bb.0:
; AVX:         vcmpneq_oqps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_one_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_oqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_one_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_oqps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ord_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ord_q:
; SSE:       # %bb.0:
; SSE:         cmpordps {{.*}}, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_ord_q:
; AVX:       # %bb.0:
; AVX:         vcmpordps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ord_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpordps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ord_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpordps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ueq_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ueq_q:
; SSE:       # %bb.0:
; SSE:         cmpeqps %xmm3, %xmm4
; SSE-NEXT:    cmpunordps %xmm3, %xmm2
; SSE-NEXT:    orps %xmm4, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_ueq_q:
; AVX:       # %bb.0:
; AVX:         vcmpeq_uqps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ueq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_uqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ueq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_uqps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ugt_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ugt_q:
; SSE:       # %bb.0:
; SSE:         ucomiss %xmm4, %xmm5
; SSE:         unpckhpd {{.*#+}} xmm5 = xmm5[1],xmm2[1]
; SSE:         unpckhpd {{.*#+}} xmm6 = xmm6[1],xmm3[1]
; SSE:         ucomiss %xmm5, %xmm6
; SSE:         ucomiss %xmm2, %xmm3
; SSE:         ucomiss %xmm2, %xmm3
;
; AVX-LABEL: test_v4f32_ugt_q:
; AVX:       # %bb.0:
; AVX:         vcmpnle_uqps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ugt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnle_uqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ugt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_uge_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_uge_q:
; SSE:       # %bb.0:
; SSE:         ucomiss %xmm4, %xmm5
; SSE:         unpckhpd {{.*#+}} xmm5 = xmm5[1],xmm2[1]
; SSE:         unpckhpd {{.*#+}} xmm6 = xmm6[1],xmm3[1]
; SSE:         ucomiss %xmm5, %xmm6
; SSE:         ucomiss %xmm2, %xmm3
; SSE:         ucomiss %xmm2, %xmm3
;
; AVX-LABEL: test_v4f32_uge_q:
; AVX:       # %bb.0:
; AVX:         vcmpnlt_uqps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_uge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnlt_uqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_uge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ult_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ult_q:
; SSE:       # %bb.0:
; SSE:         ucomiss %xmm4, %xmm5
; SSE:         unpckhpd {{.*#+}} xmm5 = xmm5[1],xmm3[1]
; SSE:         unpckhpd {{.*#+}} xmm6 = xmm6[1],xmm2[1]
; SSE:         ucomiss %xmm5, %xmm6
; SSE:         ucomiss %xmm3, %xmm2
; SSE:         ucomiss %xmm3, %xmm2
;
; AVX-LABEL: test_v4f32_ult_q:
; AVX:       # %bb.0:
; AVX:         vcmpnle_uqps {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ult_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnge_uqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ult_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqps %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ule_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ule_q:
; SSE:       # %bb.0:
; SSE:         ucomiss %xmm4, %xmm5
; SSE:         unpckhpd {{.*#+}} xmm5 = xmm5[1],xmm3[1]
; SSE:         unpckhpd {{.*#+}} xmm6 = xmm6[1],xmm2[1]
; SSE:         ucomiss %xmm5, %xmm6
; SSE:         ucomiss %xmm3, %xmm2
; SSE:         ucomiss %xmm3, %xmm2
;
; AVX-LABEL: test_v4f32_ule_q:
; AVX:       # %bb.0:
; AVX:         vcmpnlt_uqps {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ule_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngt_uqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ule_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqps %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_une_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_une_q:
; SSE:       # %bb.0:
; SSE:         cmpneqps {{.*}}, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_une_q:
; AVX:       # %bb.0:
; AVX:         vcmpneqps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_une_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneqps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_une_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneqps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_uno_q(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_uno_q:
; SSE:       # %bb.0:
; SSE:         cmpunordps {{.*}}, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_uno_q:
; AVX:       # %bb.0:
; AVX:         vcmpunordps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_uno_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunordps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_uno_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunordps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <2 x i64> @test_v2f64_oeq_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_oeq_q:
; SSE:       # %bb.0:
; SSE:         cmpeqpd {{.*}}, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_oeq_q:
; AVX:       # %bb.0:
; AVX:         vcmpeqpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_oeq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_oeq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeqpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ogt_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ogt_q:
; SSE:       # %bb.0:
; SSE:         ucomisd %xmm3, %xmm2
; SSE:         unpckhpd {{.*#+}} xmm3 = xmm3[1,1]
; SSE:         unpckhpd {{.*#+}} xmm2 = xmm2[1,1]
; SSE:         ucomisd %xmm3, %xmm2
;
; AVX-LABEL: test_v2f64_ogt_q:
; AVX:       # %bb.0:
; AVX:         vcmplt_oqpd {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ogt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgt_oqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ogt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqpd %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_oge_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_oge_q:
; SSE:       # %bb.0:
; SSE:         ucomisd %xmm3, %xmm2
; SSE:         unpckhpd {{.*#+}} xmm3 = xmm3[1,1]
; SSE:         unpckhpd {{.*#+}} xmm2 = xmm2[1,1]
; SSE:         ucomisd %xmm3, %xmm2
;
; AVX-LABEL: test_v2f64_oge_q:
; AVX:       # %bb.0:
; AVX:         vcmple_oqpd {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_oge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpge_oqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_oge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqpd %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_olt_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_olt_q:
; SSE:       # %bb.0:
; SSE:         ucomisd %xmm2, %xmm3
; SSE:         unpckhpd {{.*#+}} xmm2 = xmm2[1,1]
; SSE:         unpckhpd {{.*#+}} xmm3 = xmm3[1,1]
; SSE:         ucomisd %xmm2, %xmm3
;
; AVX-LABEL: test_v2f64_olt_q:
; AVX:       # %bb.0:
; AVX:         vcmplt_oqpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_olt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmplt_oqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_olt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ole_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ole_q:
; SSE:       # %bb.0:
; SSE:         ucomisd %xmm2, %xmm3
; SSE:         unpckhpd {{.*#+}} xmm2 = xmm2[1,1]
; SSE:         unpckhpd {{.*#+}} xmm3 = xmm3[1,1]
; SSE:         ucomisd %xmm2, %xmm3
;
; AVX-LABEL: test_v2f64_ole_q:
; AVX:       # %bb.0:
; AVX:         vcmple_oqpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ole_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmple_oqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ole_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_one_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_one_q:
; SSE:       # %bb.0:
; SSE:         cmpneqpd %xmm3, %xmm4
; SSE-NEXT:    cmpordpd %xmm3, %xmm2
; SSE-NEXT:    andpd %xmm4, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_one_q:
; AVX:       # %bb.0:
; AVX:         vcmpneq_oqpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_one_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_oqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_one_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_oqpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ord_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ord_q:
; SSE:       # %bb.0:
; SSE:         cmpordpd {{.*}}, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_ord_q:
; AVX:       # %bb.0:
; AVX:         vcmpordpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ord_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpordpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ord_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpordpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ueq_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ueq_q:
; SSE:       # %bb.0:
; SSE:         cmpeqpd %xmm3, %xmm4
; SSE-NEXT:    cmpunordpd %xmm3, %xmm2
; SSE-NEXT:    orpd %xmm4, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_ueq_q:
; AVX:       # %bb.0:
; AVX:         vcmpeq_uqpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ueq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_uqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ueq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_uqpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ugt_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ugt_q:
; SSE:       # %bb.0:
; SSE:         ucomisd %xmm2, %xmm3
; SSE:         unpckhpd {{.*#+}} xmm2 = xmm2[1,1]
; SSE:         unpckhpd {{.*#+}} xmm3 = xmm3[1,1]
; SSE:         ucomisd %xmm2, %xmm3
;
; AVX-LABEL: test_v2f64_ugt_q:
; AVX:       # %bb.0:
; AVX:         vcmpnle_uqpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ugt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnle_uqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ugt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_uge_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_uge_q:
; SSE:       # %bb.0:
; SSE:         ucomisd %xmm2, %xmm3
; SSE:         unpckhpd {{.*#+}} xmm2 = xmm2[1,1]
; SSE:         unpckhpd {{.*#+}} xmm3 = xmm3[1,1]
; SSE:         ucomisd %xmm2, %xmm3
;
; AVX-LABEL: test_v2f64_uge_q:
; AVX:       # %bb.0:
; AVX:         vcmpnlt_uqpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_uge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnlt_uqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_uge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ult_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ult_q:
; SSE:       # %bb.0:
; SSE:         ucomisd %xmm3, %xmm2
; SSE:         unpckhpd {{.*#+}} xmm3 = xmm3[1,1]
; SSE:         unpckhpd {{.*#+}} xmm2 = xmm2[1,1]
; SSE:         ucomisd %xmm3, %xmm2
;
; AVX-LABEL: test_v2f64_ult_q:
; AVX:       # %bb.0:
; AVX:         vcmpnle_uqpd {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ult_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnge_uqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ult_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqpd %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ule_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ule_q:
; SSE:       # %bb.0:
; SSE:         ucomisd %xmm3, %xmm2
; SSE:         unpckhpd {{.*#+}} xmm3 = xmm3[1,1]
; SSE:         unpckhpd {{.*#+}} xmm2 = xmm2[1,1]
; SSE:         ucomisd %xmm3, %xmm2
;
; AVX-LABEL: test_v2f64_ule_q:
; AVX:       # %bb.0:
; AVX:         vcmpnlt_uqpd {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ule_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngt_uqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ule_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqpd %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_une_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_une_q:
; SSE:       # %bb.0:
; SSE:         cmpneqpd {{.*}}, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_une_q:
; AVX:       # %bb.0:
; AVX:         vcmpneqpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_une_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneqpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_une_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneqpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_uno_q(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_uno_q:
; SSE:       # %bb.0:
; SSE:         cmpunordpd {{.*}}, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_uno_q:
; AVX:       # %bb.0:
; AVX:         vcmpunordpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_uno_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunordpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_uno_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunordpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <4 x i32> @test_v4f32_oeq_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_oeq_s:
; SSE:       # %bb.0:
; SSE:         cmpltps %xmm3, %xmm4
; SSE-NEXT:    cmpeqps %xmm3, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_oeq_s:
; AVX:       # %bb.0:
; AVX:         vcmpeq_osps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_oeq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_osps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_oeq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_osps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ogt_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ogt_s:
; SSE:       # %bb.0:
; SSE:         cmpltps {{.*}}, %xmm3
; SSE-NEXT:    andps %xmm3, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm3
; SSE-NEXT:    orps %xmm3, %xmm0
;
; AVX-LABEL: test_v4f32_ogt_s:
; AVX:       # %bb.0:
; AVX:         vcmpltps {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ogt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgtps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ogt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltps %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_oge_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_oge_s:
; SSE:       # %bb.0:
; SSE:         cmpleps {{.*}}, %xmm3
; SSE-NEXT:    andps %xmm3, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm3
; SSE-NEXT:    orps %xmm3, %xmm0
;
; AVX-LABEL: test_v4f32_oge_s:
; AVX:       # %bb.0:
; AVX:         vcmpleps {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_oge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgeps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_oge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpleps %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_olt_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_olt_s:
; SSE:       # %bb.0:
; SSE:         cmpltps {{.*}}, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_olt_s:
; AVX:       # %bb.0:
; AVX:         vcmpltps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_olt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpltps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_olt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ole_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ole_s:
; SSE:       # %bb.0:
; SSE:         cmpleps {{.*}}, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_ole_s:
; AVX:       # %bb.0:
; AVX:         vcmpleps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ole_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpleps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ole_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpleps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_one_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_one_s:
; SSE:       # %bb.0:
; SSE:         cmpltps %xmm3, %xmm4
; SSE:         cmpneqps %xmm3, %xmm4
; SSE-NEXT:    cmpordps %xmm3, %xmm2
; SSE-NEXT:    andps %xmm4, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_one_s:
; AVX:       # %bb.0:
; AVX:         vcmpneq_osps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_one_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_osps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_one_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_osps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ord_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ord_s:
; SSE:       # %bb.0:
; SSE:         cmpltps %xmm3, %xmm4
; SSE-NEXT:    cmpordps %xmm3, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_ord_s:
; AVX:       # %bb.0:
; AVX:         vcmpord_sps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ord_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpord_sps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ord_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpord_sps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ueq_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ueq_s:
; SSE:       # %bb.0:
; SSE:         cmpltps %xmm3, %xmm4
; SSE:         cmpeqps %xmm3, %xmm4
; SSE-NEXT:    cmpunordps %xmm3, %xmm2
; SSE-NEXT:    orps %xmm4, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_ueq_s:
; AVX:       # %bb.0:
; AVX:         vcmpeq_usps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ueq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_usps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ueq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_usps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ugt_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ugt_s:
; SSE:       # %bb.0:
; SSE:         cmpnleps {{.*}}, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_ugt_s:
; AVX:       # %bb.0:
; AVX:         vcmpnleps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ugt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnleps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ugt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnleps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_uge_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_uge_s:
; SSE:       # %bb.0:
; SSE:         cmpnltps {{.*}}, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_uge_s:
; AVX:       # %bb.0:
; AVX:         vcmpnltps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_uge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnltps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_uge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ult_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ult_s:
; SSE:       # %bb.0:
; SSE:         cmpnleps {{.*}}, %xmm3
; SSE-NEXT:    andps %xmm3, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm3
; SSE-NEXT:    orps %xmm3, %xmm0
;
; AVX-LABEL: test_v4f32_ult_s:
; AVX:       # %bb.0:
; AVX:         vcmpnleps {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ult_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngeps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ult_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnleps %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_ule_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_ule_s:
; SSE:       # %bb.0:
; SSE:         cmpnltps {{.*}}, %xmm3
; SSE-NEXT:    andps %xmm3, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm3
; SSE-NEXT:    orps %xmm3, %xmm0
;
; AVX-LABEL: test_v4f32_ule_s:
; AVX:       # %bb.0:
; AVX:         vcmpnltps {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_ule_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngtps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_ule_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltps %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_une_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_une_s:
; SSE:       # %bb.0:
; SSE:         cmpltps %xmm3, %xmm4
; SSE-NEXT:    cmpneqps %xmm3, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_une_s:
; AVX:       # %bb.0:
; AVX:         vcmpneq_usps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_une_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_usps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_une_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_usps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <4 x i32> @test_v4f32_uno_s(<4 x i32> %a, <4 x i32> %b, <4 x float> %f1, <4 x float> %f2) #0 {
; SSE-LABEL: test_v4f32_uno_s:
; SSE:       # %bb.0:
; SSE:         cmpltps %xmm3, %xmm4
; SSE-NEXT:    cmpunordps %xmm3, %xmm2
; SSE-NEXT:    andps %xmm2, %xmm0
; SSE-NEXT:    andnps %xmm1, %xmm2
; SSE-NEXT:    orps %xmm2, %xmm0
;
; AVX-LABEL: test_v4f32_uno_s:
; AVX:       # %bb.0:
; AVX:         vcmpunord_sps {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvps %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v4f32_uno_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunord_sps 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v4f32_uno_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunord_sps %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmd %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %f1, <4 x float> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %res
}

define <2 x i64> @test_v2f64_oeq_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_oeq_s:
; SSE:       # %bb.0:
; SSE:         cmpltpd %xmm3, %xmm4
; SSE-NEXT:    cmpeqpd %xmm3, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_oeq_s:
; AVX:       # %bb.0:
; AVX:         vcmpeq_ospd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_oeq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_ospd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_oeq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_ospd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ogt_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ogt_s:
; SSE:       # %bb.0:
; SSE:         cmpltpd {{.*}}, %xmm3
; SSE-NEXT:    andpd %xmm3, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm3
; SSE-NEXT:    orpd %xmm3, %xmm0
;
; AVX-LABEL: test_v2f64_ogt_s:
; AVX:       # %bb.0:
; AVX:         vcmpltpd {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ogt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgtpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ogt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltpd %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_oge_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_oge_s:
; SSE:       # %bb.0:
; SSE:         cmplepd {{.*}}, %xmm3
; SSE-NEXT:    andpd %xmm3, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm3
; SSE-NEXT:    orpd %xmm3, %xmm0
;
; AVX-LABEL: test_v2f64_oge_s:
; AVX:       # %bb.0:
; AVX:         vcmplepd {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_oge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgepd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_oge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplepd %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_olt_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_olt_s:
; SSE:       # %bb.0:
; SSE:         cmpltpd {{.*}}, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_olt_s:
; AVX:       # %bb.0:
; AVX:         vcmpltpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_olt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpltpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_olt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ole_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ole_s:
; SSE:       # %bb.0:
; SSE:         cmplepd {{.*}}, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_ole_s:
; AVX:       # %bb.0:
; AVX:         vcmplepd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ole_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmplepd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ole_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplepd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_one_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_one_s:
; SSE:       # %bb.0:
; SSE:         cmpltpd %xmm3, %xmm4
; SSE:         cmpneqpd %xmm3, %xmm4
; SSE-NEXT:    cmpordpd %xmm3, %xmm2
; SSE-NEXT:    andpd %xmm4, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_one_s:
; AVX:       # %bb.0:
; AVX:         vcmpneq_ospd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_one_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_ospd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_one_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_ospd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ord_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ord_s:
; SSE:       # %bb.0:
; SSE:         cmpltpd %xmm3, %xmm4
; SSE-NEXT:    cmpordpd %xmm3, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_ord_s:
; AVX:       # %bb.0:
; AVX:         vcmpord_spd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ord_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpord_spd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ord_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpord_spd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ueq_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ueq_s:
; SSE:       # %bb.0:
; SSE:         cmpltpd %xmm3, %xmm4
; SSE:         cmpeqpd %xmm3, %xmm4
; SSE-NEXT:    cmpunordpd %xmm3, %xmm2
; SSE-NEXT:    orpd %xmm4, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_ueq_s:
; AVX:       # %bb.0:
; AVX:         vcmpeq_uspd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ueq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_uspd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ueq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_uspd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ugt_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ugt_s:
; SSE:       # %bb.0:
; SSE:         cmpnlepd {{.*}}, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_ugt_s:
; AVX:       # %bb.0:
; AVX:         vcmpnlepd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ugt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnlepd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ugt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlepd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_uge_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_uge_s:
; SSE:       # %bb.0:
; SSE:         cmpnltpd {{.*}}, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_uge_s:
; AVX:       # %bb.0:
; AVX:         vcmpnltpd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_uge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnltpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_uge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltpd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ult_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ult_s:
; SSE:       # %bb.0:
; SSE:         cmpnlepd {{.*}}, %xmm3
; SSE-NEXT:    andpd %xmm3, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm3
; SSE-NEXT:    orpd %xmm3, %xmm0
;
; AVX-LABEL: test_v2f64_ult_s:
; AVX:       # %bb.0:
; AVX:         vcmpnlepd {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ult_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngepd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ult_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlepd %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_ule_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_ule_s:
; SSE:       # %bb.0:
; SSE:         cmpnltpd {{.*}}, %xmm3
; SSE-NEXT:    andpd %xmm3, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm3
; SSE-NEXT:    orpd %xmm3, %xmm0
;
; AVX-LABEL: test_v2f64_ule_s:
; AVX:       # %bb.0:
; AVX:         vcmpnltpd {{.*}}, %xmm3, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_ule_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngtpd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_ule_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltpd %xmm2, %xmm3, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_une_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_une_s:
; SSE:       # %bb.0:
; SSE:         cmpltpd %xmm3, %xmm4
; SSE-NEXT:    cmpneqpd %xmm3, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_une_s:
; AVX:       # %bb.0:
; AVX:         vcmpneq_uspd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_une_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_uspd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_une_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_uspd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

define <2 x i64> @test_v2f64_uno_s(<2 x i64> %a, <2 x i64> %b, <2 x double> %f1, <2 x double> %f2) #0 {
; SSE-LABEL: test_v2f64_uno_s:
; SSE:       # %bb.0:
; SSE:         cmpltpd %xmm3, %xmm4
; SSE-NEXT:    cmpunordpd %xmm3, %xmm2
; SSE-NEXT:    andpd %xmm2, %xmm0
; SSE-NEXT:    andnpd %xmm1, %xmm2
; SSE-NEXT:    orpd %xmm2, %xmm0
;
; AVX-LABEL: test_v2f64_uno_s:
; AVX:       # %bb.0:
; AVX:         vcmpunord_spd {{.*}}, %xmm2, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
;
; AVX512-32-LABEL: test_v2f64_uno_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunord_spd 8(%ebp), %xmm2, %k1
; AVX512-32-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
;
; AVX512-64-LABEL: test_v2f64_uno_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunord_spd %xmm3, %xmm2, %k1
; AVX512-64-NEXT:    vpblendmq %xmm0, %xmm1, %xmm0 {%k1}
  %cond = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %f1, <2 x double> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %res
}

attributes #0 = { strictfp }

declare <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float>, <4 x float>, metadata, metadata)
declare <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double>, <2 x double>, metadata, metadata)
declare <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float>, <4 x float>, metadata, metadata)
declare <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double>, <2 x double>, metadata, metadata)
