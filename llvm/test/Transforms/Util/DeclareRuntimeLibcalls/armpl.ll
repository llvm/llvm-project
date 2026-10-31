; REQUIRES: aarch64-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-unknown-linux -mattr=+neon,+sve -vector-library=ArmPL < %s | FileCheck %s

; CHECK: declare <vscale x 4 x float> @armpl_svacos_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS:#[0-9]+]]

; CHECK: declare <vscale x 2 x double> @armpl_svacos_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svacosh_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svacosh_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svasin_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svasin_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svasinh_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svasinh_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svatan2_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svatan2_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svatan_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svatan_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svatanh_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svatanh_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svcbrt_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svcbrt_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare void @armpl_svcexpi_f32_x(...)

; CHECK: declare void @armpl_svcexpi_f64_x(...)

; CHECK: declare void @armpl_svcexpipi_f32_x(...)

; CHECK: declare void @armpl_svcexpipi_f64_x(...)

; CHECK: declare <vscale x 4 x float> @armpl_svcopysign_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svcopysign_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svcos_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svcos_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svcosh_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svcosh_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svcospi_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svcospi_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_sverf_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_sverf_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_sverfc_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_sverfc_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svexp10_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svexp10_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svexp2_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svexp2_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svexp_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svexp_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svexpm1_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svexpm1_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svfdim_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svfdim_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svfma_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svfma_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svfmax_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svfmax_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svfmin_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svfmin_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svfmod_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svfmod_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svhypot_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svhypot_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x i32> @armpl_svilogb_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x i32> @armpl_svilogb_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svldexp_f32_x(<vscale x 4 x float>, <vscale x 4 x i32>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svldexp_f64_x(<vscale x 2 x double>, <vscale x 2 x i32>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svlgamma_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svlgamma_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svlog10_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svlog10_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svlog1p_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svlog1p_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svlog2_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svlog2_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svlog_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svlog_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare void @armpl_svmodf_f32_stret_x(...)

; CHECK: declare <vscale x 4 x float> @armpl_svmodf_f32_x(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, <vscale x 4 x i1>) [[ATTRS_PTR_ARG:#[0-9]+]]

; CHECK: declare void @armpl_svmodf_f64_stret_x(...)

; CHECK: declare <vscale x 2 x double> @armpl_svmodf_f64_x(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, <vscale x 2 x i1>) [[ATTRS_PTR_ARG]]

; CHECK: declare <vscale x 4 x float> @armpl_svnextafter_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svnextafter_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svpow_f32_x(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svpow_f64_x(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svsin_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svsin_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare void @armpl_svsincos_f32_x(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16, <vscale x 4 x i1>) [[ATTRS_PTR_ARG]]

; CHECK: declare void @armpl_svsincos_f64_x(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16, <vscale x 2 x i1>) [[ATTRS_PTR_ARG]]

; CHECK: declare void @armpl_svsincospi_f32_x(<vscale x 4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16, <vscale x 4 x i1>) [[ATTRS_PTR_ARG]]

; CHECK: declare void @armpl_svsincospi_f64_x(<vscale x 2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16, <vscale x 2 x i1>) [[ATTRS_PTR_ARG]]

; CHECK: declare <vscale x 4 x float> @armpl_svsinh_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svsinh_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svsinpi_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svsinpi_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svsqrt_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svsqrt_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svtan_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svtan_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svtanh_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svtanh_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 4 x float> @armpl_svtgamma_f32_x(<vscale x 4 x float>, <vscale x 4 x i1>) [[ATTRS]]

; CHECK: declare <vscale x 2 x double> @armpl_svtgamma_f64_x(<vscale x 2 x double>, <vscale x 2 x i1>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vacoshq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vacoshq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vacosq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vacosq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vasinhq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vasinhq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vasinq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vasinq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vatan2q_f32(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vatan2q_f64(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vatanhq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vatanhq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vatanq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vatanq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vcbrtq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vcbrtq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs void @armpl_vcexpipiq_f32(...)

; CHECK: declare aarch64_vector_pcs void @armpl_vcexpipiq_f64(...)

; CHECK: declare aarch64_vector_pcs void @armpl_vcexpiq_f32(...)

; CHECK: declare aarch64_vector_pcs void @armpl_vcexpiq_f64(...)

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vcopysignq_f32(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vcopysignq_f64(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vcoshq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vcoshq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vcospiq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vcospiq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vcosq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vcosq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_verfcq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_verfcq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_verfq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_verfq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vexp10q_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vexp10q_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vexp2q_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vexp2q_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vexpm1q_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vexpm1q_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vexpq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vexpq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vfdimq_f32(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vfdimq_f64(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vfmaq_f32(<4 x float>, <4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vfmaq_f64(<2 x double>, <2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vfmaxq_f32(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vfmaxq_f64(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vfminq_f32(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vfminq_f64(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vfmodq_f32(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vfmodq_f64(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vhypotq_f32(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vhypotq_f64(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x i32> @armpl_vilogbq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x i32> @armpl_vilogbq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vldexpq_f32(<4 x float>, <4 x i32>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vldexpq_f64(<2 x double>, <2 x i32>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vlgammaq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vlgammaq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vlog10q_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vlog10q_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vlog1pq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vlog1pq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vlog2q_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vlog2q_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vlogq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vlogq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vmodfq_f32(<4 x float>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare aarch64_vector_pcs void @armpl_vmodfq_f32_stret(...)

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vmodfq_f64(<2 x double>, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare aarch64_vector_pcs void @armpl_vmodfq_f64_stret(...)

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vnextafterq_f32(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vnextafterq_f64(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vpowq_f32(<4 x float>, <4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vpowq_f64(<2 x double>, <2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs void @armpl_vsincospiq_f32(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare aarch64_vector_pcs void @armpl_vsincospiq_f64(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare aarch64_vector_pcs void @armpl_vsincosq_f32(<4 x float>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare aarch64_vector_pcs void @armpl_vsincosq_f64(<2 x double>, ptr noalias nonnull writeonly align 16, ptr noalias nonnull writeonly align 16) [[ATTRS_PTR_ARG]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vsinhq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vsinhq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vsinpiq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vsinpiq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vsinq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vsinq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vsqrtq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vsqrtq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vtanhq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vtanhq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vtanq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vtanq_f64(<2 x double>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <4 x float> @armpl_vtgammaq_f32(<4 x float>) [[ATTRS]]

; CHECK: declare aarch64_vector_pcs <2 x double> @armpl_vtgammaq_f64(<2 x double>) [[ATTRS]]

; CHECK: attributes [[ATTRS]] = { nocallback nofree nosync nounwind willreturn }
; CHECK: attributes [[ATTRS_PTR_ARG]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
