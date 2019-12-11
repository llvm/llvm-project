; RUN: llc -disable-strictnode-mutation < %s -mtriple=i686-unknown-unknown -mattr=+avx -O3 | FileCheck %s --check-prefixes=CHECK,AVX
; RUN: llc -disable-strictnode-mutation < %s -mtriple=x86_64-unknown-unknown -mattr=+avx -O3 | FileCheck %s --check-prefixes=CHECK,AVX
; RUN: llc -disable-strictnode-mutation < %s -mtriple=i686-unknown-unknown -mattr=+avx512f -mattr=+avx512vl -O3 | FileCheck %s --check-prefixes=CHECK,AVX512-32
; RUN: llc -disable-strictnode-mutation < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512f -mattr=+avx512vl -O3 | FileCheck %s --check-prefixes=CHECK,AVX512-64

define <8 x i32> @test_v8f32_oeq_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_oeq_q:
; AVX:       # %bb.0:
; AVX:         vcmpeqps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_oeq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_oeq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeqps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ogt_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ogt_q:
; AVX:       # %bb.0:
; AVX:         vcmplt_oqps {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ogt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgt_oqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ogt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqps %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_oge_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_oge_q:
; AVX:       # %bb.0:
; AVX:         vcmple_oqps {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_oge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpge_oqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_oge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqps %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_olt_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_olt_q:
; AVX:       # %bb.0:
; AVX:         vcmplt_oqps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_olt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmplt_oqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_olt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ole_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ole_q:
; AVX:       # %bb.0:
; AVX:         vcmple_oqps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ole_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmple_oqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ole_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_one_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_one_q:
; AVX:       # %bb.0:
; AVX:         vcmpneq_oqps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_one_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_oqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_one_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_oqps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ord_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ord_q:
; AVX:       # %bb.0:
; AVX:         vcmpordps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ord_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpordps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ord_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpordps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ueq_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ueq_q:
; AVX:       # %bb.0:
; AVX:         vcmpeq_uqps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ueq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_uqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ueq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_uqps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ugt_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ugt_q:
; AVX:       # %bb.0:
; AVX:         vcmpnle_uqps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ugt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnle_uqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ugt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_uge_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_uge_q:
; AVX:       # %bb.0:
; AVX:         vcmpnlt_uqps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_uge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnlt_uqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_uge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ult_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ult_q:
; AVX:       # %bb.0:
; AVX:         vcmpnle_uqps {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ult_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnge_uqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ult_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqps %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ule_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ule_q:
; AVX:       # %bb.0:
; AVX:         vcmpnlt_uqps {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ule_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngt_uqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ule_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqps %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_une_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_une_q:
; AVX:       # %bb.0:
; AVX:         vcmpneqps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_une_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneqps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_une_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneqps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_uno_q(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_uno_q:
; AVX:       # %bb.0:
; AVX:         vcmpunordps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_uno_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunordps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_uno_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunordps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <4 x i64> @test_v4f64_oeq_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_oeq_q:
; AVX:       # %bb.0:
; AVX:         vcmpeqpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_oeq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_oeq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeqpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ogt_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ogt_q:
; AVX:       # %bb.0:
; AVX:         vcmplt_oqpd {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ogt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgt_oqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ogt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqpd %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_oge_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_oge_q:
; AVX:       # %bb.0:
; AVX:         vcmple_oqpd {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_oge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpge_oqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_oge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqpd %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_olt_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_olt_q:
; AVX:       # %bb.0:
; AVX:         vcmplt_oqpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_olt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmplt_oqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_olt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ole_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ole_q:
; AVX:       # %bb.0:
; AVX:         vcmple_oqpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ole_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmple_oqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ole_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_one_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_one_q:
; AVX:       # %bb.0:
; AVX:         vcmpneq_oqpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_one_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_oqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_one_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_oqpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ord_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ord_q:
; AVX:       # %bb.0:
; AVX:         vcmpordpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ord_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpordpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ord_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpordpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ueq_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ueq_q:
; AVX:       # %bb.0:
; AVX:         vcmpeq_uqpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ueq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_uqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ueq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_uqpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ugt_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ugt_q:
; AVX:       # %bb.0:
; AVX:         vcmpnle_uqpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ugt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnle_uqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ugt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_uge_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_uge_q:
; AVX:       # %bb.0:
; AVX:         vcmpnlt_uqpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_uge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnlt_uqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_uge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ult_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ult_q:
; AVX:       # %bb.0:
; AVX:         vcmpnle_uqpd {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ult_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnge_uqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ult_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqpd %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ule_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ule_q:
; AVX:       # %bb.0:
; AVX:         vcmpnlt_uqpd {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ule_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngt_uqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ule_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqpd %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_une_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_une_q:
; AVX:       # %bb.0:
; AVX:         vcmpneqpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_une_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneqpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_une_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneqpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_uno_q(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_uno_q:
; AVX:       # %bb.0:
; AVX:         vcmpunordpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_uno_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunordpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_uno_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunordpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <8 x i32> @test_v8f32_oeq_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_oeq_s:
; AVX:       # %bb.0:
; AVX:         vcmpeq_osps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_oeq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_osps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_oeq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_osps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ogt_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ogt_s:
; AVX:       # %bb.0:
; AVX:         vcmpltps {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ogt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgtps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ogt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltps %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_oge_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_oge_s:
; AVX:       # %bb.0:
; AVX:         vcmpleps {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_oge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgeps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_oge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpleps %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_olt_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_olt_s:
; AVX:       # %bb.0:
; AVX:         vcmpltps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_olt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpltps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_olt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ole_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ole_s:
; AVX:       # %bb.0:
; AVX:         vcmpleps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ole_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpleps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ole_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpleps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_one_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_one_s:
; AVX:       # %bb.0:
; AVX:         vcmpneq_osps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_one_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_osps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_one_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_osps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ord_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ord_s:
; AVX:       # %bb.0:
; AVX:         vcmpord_sps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ord_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpord_sps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ord_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpord_sps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ueq_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ueq_s:
; AVX:       # %bb.0:
; AVX:         vcmpeq_usps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ueq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_usps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ueq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_usps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ugt_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ugt_s:
; AVX:       # %bb.0:
; AVX:         vcmpnleps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ugt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnleps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ugt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnleps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_uge_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_uge_s:
; AVX:       # %bb.0:
; AVX:         vcmpnltps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_uge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnltps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_uge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ult_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ult_s:
; AVX:       # %bb.0:
; AVX:         vcmpnleps {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ult_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngeps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ult_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnleps %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_ule_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_ule_s:
; AVX:       # %bb.0:
; AVX:         vcmpnltps {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_ule_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngtps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_ule_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltps %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_une_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_une_s:
; AVX:       # %bb.0:
; AVX:         vcmpneq_usps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_une_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_usps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_une_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_usps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <8 x i32> @test_v8f32_uno_s(<8 x i32> %a, <8 x i32> %b, <8 x float> %f1, <8 x float> %f2) #0 {
; AVX-LABEL: test_v8f32_uno_s:
; AVX:       # %bb.0:
; AVX:         vcmpunord_sps {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvps %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v8f32_uno_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunord_sps 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v8f32_uno_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunord_sps %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmd %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(
                                               <8 x float> %f1, <8 x float> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %res
}

define <4 x i64> @test_v4f64_oeq_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_oeq_s:
; AVX:       # %bb.0:
; AVX:         vcmpeq_ospd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_oeq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_ospd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_oeq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_ospd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ogt_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ogt_s:
; AVX:       # %bb.0:
; AVX:         vcmpltpd {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ogt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgtpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ogt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltpd %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_oge_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_oge_s:
; AVX:       # %bb.0:
; AVX:         vcmplepd {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_oge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgepd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_oge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplepd %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_olt_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_olt_s:
; AVX:       # %bb.0:
; AVX:         vcmpltpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_olt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpltpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_olt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ole_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ole_s:
; AVX:       # %bb.0:
; AVX:         vcmplepd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ole_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmplepd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ole_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplepd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_one_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_one_s:
; AVX:       # %bb.0:
; AVX:         vcmpneq_ospd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_one_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_ospd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_one_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_ospd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ord_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ord_s:
; AVX:       # %bb.0:
; AVX:         vcmpord_spd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ord_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpord_spd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ord_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpord_spd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ueq_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ueq_s:
; AVX:       # %bb.0:
; AVX:         vcmpeq_uspd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ueq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_uspd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ueq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_uspd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ugt_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ugt_s:
; AVX:       # %bb.0:
; AVX:         vcmpnlepd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ugt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnlepd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ugt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlepd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_uge_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_uge_s:
; AVX:       # %bb.0:
; AVX:         vcmpnltpd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_uge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnltpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_uge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltpd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ult_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ult_s:
; AVX:       # %bb.0:
; AVX:         vcmpnlepd {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ult_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngepd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ult_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlepd %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_ule_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_ule_s:
; AVX:       # %bb.0:
; AVX:         vcmpnltpd {{.*}}, %ymm3, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_ule_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngtpd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_ule_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltpd %ymm2, %ymm3, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_une_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_une_s:
; AVX:       # %bb.0:
; AVX:         vcmpneq_uspd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_une_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_uspd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_une_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_uspd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

define <4 x i64> @test_v4f64_uno_s(<4 x i64> %a, <4 x i64> %b, <4 x double> %f1, <4 x double> %f2) #0 {
; AVX-LABEL: test_v4f64_uno_s:
; AVX:       # %bb.0:
; AVX:         vcmpunord_spd {{.*}}, %ymm2, %ymm2
; AVX-NEXT:    vblendvpd %ymm2, %ymm0, %ymm1, %ymm0
;
; AVX512-32-LABEL: test_v4f64_uno_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunord_spd 8(%ebp), %ymm2, %k1
; AVX512-32-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
;
; AVX512-64-LABEL: test_v4f64_uno_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunord_spd %ymm3, %ymm2, %k1
; AVX512-64-NEXT:    vpblendmq %ymm0, %ymm1, %ymm0 {%k1}
  %cond = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(
                                               <4 x double> %f1, <4 x double> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <4 x i1> %cond, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %res
}

attributes #0 = { strictfp }

declare <8 x i1> @llvm.experimental.constrained.fcmp.v8f32(<8 x float>, <8 x float>, metadata, metadata)
declare <4 x i1> @llvm.experimental.constrained.fcmp.v4f64(<4 x double>, <4 x double>, metadata, metadata)
declare <8 x i1> @llvm.experimental.constrained.fcmps.v8f32(<8 x float>, <8 x float>, metadata, metadata)
declare <4 x i1> @llvm.experimental.constrained.fcmps.v4f64(<4 x double>, <4 x double>, metadata, metadata)
