; RUN: llc -disable-strictnode-mutation < %s -mtriple=i686-unknown-unknown -mattr=+avx512f -O3 | FileCheck %s --check-prefixes=CHECK,AVX512-32
; RUN: llc -disable-strictnode-mutation < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512f -O3 | FileCheck %s --check-prefixes=CHECK,AVX512-64

define <16 x i32> @test_v16f32_oeq_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_oeq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_oeq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeqps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ogt_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ogt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgt_oqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ogt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqps %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_oge_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_oge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpge_oqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_oge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqps %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_olt_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_olt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmplt_oqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_olt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ole_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ole_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmple_oqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ole_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_one_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_one_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_oqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_one_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_oqps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ord_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ord_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpordps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ord_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpordps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ueq_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ueq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_uqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ueq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_uqps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ugt_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ugt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnle_uqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ugt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_uge_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_uge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnlt_uqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_uge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ult_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ult_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnge_uqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ult_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqps %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ule_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ule_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngt_uqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ule_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqps %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_une_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_une_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneqps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_une_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneqps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_uno_q(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_uno_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunordps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_uno_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunordps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <8 x i64> @test_v8f64_oeq_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_oeq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_oeq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeqpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ogt_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ogt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgt_oqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ogt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqpd %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_oge_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_oge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpge_oqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_oge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqpd %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_olt_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_olt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmplt_oqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_olt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplt_oqpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ole_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ole_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmple_oqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ole_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmple_oqpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_one_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_one_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_oqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_one_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_oqpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ord_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ord_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpordpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ord_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpordpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ueq_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ueq_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_uqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ueq_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_uqpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ugt_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ugt_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnle_uqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ugt_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_uge_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_uge_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnlt_uqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_uge_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ult_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ult_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnge_uqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ult_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnle_uqpd %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ule_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ule_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngt_uqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ule_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlt_uqpd %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_une_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_une_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneqpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_une_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneqpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_uno_q(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_uno_q:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunordpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_uno_q:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunordpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <16 x i32> @test_v16f32_oeq_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_oeq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_osps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_oeq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_osps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ogt_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ogt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgtps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ogt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltps %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_oge_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_oge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgeps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_oge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpleps %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_olt_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_olt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpltps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_olt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ole_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ole_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpleps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ole_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpleps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_one_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_one_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_osps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_one_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_osps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ord_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ord_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpord_sps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ord_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpord_sps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ueq_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ueq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_usps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ueq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_usps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ugt_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ugt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnleps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ugt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnleps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_uge_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_uge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnltps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_uge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ult_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ult_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngeps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ult_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnleps %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_ule_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_ule_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngtps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_ule_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltps %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_une_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_une_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_usps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_une_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_usps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <16 x i32> @test_v16f32_uno_s(<16 x i32> %a, <16 x i32> %b, <16 x float> %f1, <16 x float> %f2) #0 {
; AVX512-32-LABEL: test_v16f32_uno_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunord_sps 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v16f32_uno_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunord_sps %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmd %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(
                                               <16 x float> %f1, <16 x float> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

define <8 x i64> @test_v8f64_oeq_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_oeq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_ospd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_oeq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_ospd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ogt_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ogt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgtpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ogt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltpd %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_oge_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_oge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpgepd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_oge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplepd %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"oge",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_olt_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_olt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpltpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_olt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpltpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ole_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ole_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmplepd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ole_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmplepd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_one_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_one_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_ospd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_one_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_ospd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"one",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ord_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ord_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpord_spd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ord_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpord_spd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ord",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ueq_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ueq_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpeq_uspd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ueq_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpeq_uspd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ugt_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ugt_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnlepd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ugt_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlepd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ugt",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_uge_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_uge_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpnltpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_uge_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltpd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"uge",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ult_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ult_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngepd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ult_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnlepd %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_ule_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_ule_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpngtpd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_ule_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpnltpd %zmm2, %zmm3, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"ule",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_une_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_une_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpneq_uspd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_une_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpneq_uspd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"une",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

define <8 x i64> @test_v8f64_uno_s(<8 x i64> %a, <8 x i64> %b, <8 x double> %f1, <8 x double> %f2) #0 {
; AVX512-32-LABEL: test_v8f64_uno_s:
; AVX512-32:       # %bb.0:
; AVX512-32:         vcmpunord_spd 8(%ebp), %zmm2, %k1
; AVX512-32-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
;
; AVX512-64-LABEL: test_v8f64_uno_s:
; AVX512-64:       # %bb.0:
; AVX512-64-NEXT:    vcmpunord_spd %zmm3, %zmm2, %k1
; AVX512-64-NEXT:    vpblendmq %zmm0, %zmm1, %zmm0 {%k1}
  %cond = call <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(
                                               <8 x double> %f1, <8 x double> %f2, metadata !"uno",
                                               metadata !"fpexcept.strict") #0
  %res = select <8 x i1> %cond, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %res
}

attributes #0 = { strictfp }

declare <16 x i1> @llvm.experimental.constrained.fcmp.v16f32(<16 x float>, <16 x float>, metadata, metadata)
declare <8 x i1> @llvm.experimental.constrained.fcmp.v8f64(<8 x double>, <8 x double>, metadata, metadata)
declare <16 x i1> @llvm.experimental.constrained.fcmps.v16f32(<16 x float>, <16 x float>, metadata, metadata)
declare <8 x i1> @llvm.experimental.constrained.fcmps.v8f64(<8 x double>, <8 x double>, metadata, metadata)
