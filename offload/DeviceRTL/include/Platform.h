//===--------- Platform.h - OpenMP target specific declarations --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_PLATFORM_H
#define OMPTARGET_PLATFORM_H

namespace platform {

#pragma omp begin declare target device_type(nohost)

// We cannot use an OpenMP variant because we require "C" linkage.
#ifdef __AMDGPU__

// The ROCm device library uses control globals to alter codegen for the
// different targets. To avoid needing to link them in manually we simply
// define them here.
extern "C" {

// Disable unsafe math optimizations in the implementation.
extern const inline bool __oclc_unsafe_math_opt = 0;

// Disable denormalization at zero optimizations in the implementation.
extern const inline bool __oclc_daz_opt = 0;

// Disable rounding optimizations for 32-bit square roots.
extern const inline bool __oclc_correctly_rounded_sqrt32 = 1;

// Disable finite math optimizations.
extern const inline bool __oclc_finite_only_opt = 0;

// Spoof this to wave64 since we only compile for a single architecture.
extern const inline bool __oclc_wavefrontsize64 = 1;

#if defined(__gfx700__)
extern const inline unsigned __oclc_ISA_version = 7000;
#elif defined(__gfx701__)
extern const inline unsigned __oclc_ISA_version = 7001;
#elif defined(__gfx702__)
extern const inline unsigned __oclc_ISA_version = 7002;
#elif defined(__gfx703__)
extern const inline unsigned __oclc_ISA_version = 7003;
#elif defined(__gfx704__)
extern const inline unsigned __oclc_ISA_version = 7004;
#elif defined(__gfx705__)
extern const inline unsigned __oclc_ISA_version = 7005;
#elif defined(__gfx801__)
extern const inline unsigned __oclc_ISA_version = 8001;
#elif defined(__gfx802__)
extern const inline unsigned __oclc_ISA_version = 8002;
#elif defined(__gfx803__)
extern const inline unsigned __oclc_ISA_version = 8003;
#elif defined(__gfx805__)
extern const inline unsigned __oclc_ISA_version = 8005;
#elif defined(__gfx810__)
extern const inline unsigned __oclc_ISA_version = 8100;
#elif defined(__gfx900__)
extern const inline unsigned __oclc_ISA_version = 9000;
#elif defined(__gfx902__)
extern const inline unsigned __oclc_ISA_version = 9002;
#elif defined(__gfx904__)
extern const inline unsigned __oclc_ISA_version = 9004;
#elif defined(__gfx906__)
extern const inline unsigned __oclc_ISA_version = 9006;
#elif defined(__gfx908__)
extern const inline unsigned __oclc_ISA_version = 9008;
#elif defined(__gfx909__)
extern const inline unsigned __oclc_ISA_version = 9009;
#elif defined(__gfx90a__)
extern const inline unsigned __oclc_ISA_version = 9010;
#elif defined(__gfx90c__)
extern const inline unsigned __oclc_ISA_version = 9012;
#elif defined(__gfx942__)
extern const inline unsigned __oclc_ISA_version = 9402;
#elif defined(__gfx950__)
extern const inline unsigned __oclc_ISA_version = 9500;
#elif defined(__gfx1010__)
extern const inline unsigned __oclc_ISA_version = 10100;
#elif defined(__gfx1011__)
extern const inline unsigned __oclc_ISA_version = 10101;
#elif defined(__gfx1012__)
extern const inline unsigned __oclc_ISA_version = 10102;
#elif defined(__gfx1013__)
extern const inline unsigned __oclc_ISA_version = 10103;
#elif defined(__gfx1030__)
extern const inline unsigned __oclc_ISA_version = 10300;
#elif defined(__gfx1031__)
extern const inline unsigned __oclc_ISA_version = 10301;
#elif defined(__gfx1032__)
extern const inline unsigned __oclc_ISA_version = 10302;
#elif defined(__gfx1033__)
extern const inline unsigned __oclc_ISA_version = 10303;
#elif defined(__gfx1034__)
extern const inline unsigned __oclc_ISA_version = 10304;
#elif defined(__gfx1035__)
extern const inline unsigned __oclc_ISA_version = 10305;
#elif defined(__gfx1036__)
extern const inline unsigned __oclc_ISA_version = 10306;
#elif defined(__gfx1100__)
extern const inline unsigned __oclc_ISA_version = 11000;
#elif defined(__gfx1101__)
extern const inline unsigned __oclc_ISA_version = 11001;
#elif defined(__gfx1102__)
extern const inline unsigned __oclc_ISA_version = 11002;
#elif defined(__gfx1103__)
extern const inline unsigned __oclc_ISA_version = 11003;
#elif defined(__gfx1150__)
extern const inline unsigned __oclc_ISA_version = 11500;
#elif defined(__gfx1151__)
extern const inline unsigned __oclc_ISA_version = 11501;
#elif defined(__gfx1152__)
extern const inline unsigned __oclc_ISA_version = 11502;
#elif defined(__gfx1153__)
extern const inline unsigned __oclc_ISA_version = 11503;
#elif defined(__gfx1200__)
extern const inline unsigned __oclc_ISA_version = 12000;
#elif defined(__gfx1201__)
extern const inline unsigned __oclc_ISA_version = 12001;
#elif defined(__gfx9_generic__)
extern const inline unsigned __oclc_ISA_version = 9000;
#elif defined(__gfx9_4_generic__)
extern const inline unsigned __oclc_ISA_version = 9402;
#elif defined(__gfx10_1_generic__)
extern const inline unsigned __oclc_ISA_version = 10100;
#elif defined(__gfx10_3_generic__)
extern const inline unsigned __oclc_ISA_version = 10300;
#elif defined(__gfx11_generic__)
extern const inline unsigned __oclc_ISA_version = 11003;
#elif defined(__gfx12_generic__)
extern const inline unsigned __oclc_ISA_version = 12000;
#else
// The only thing this controls that we care about is fast FMA.
// FIXME: We need to stop relying on the DeviceRTL math libs this way.
extern const inline unsigned __oclc_ISA_version = 7001;
#endif
}

// These aliases cause clang to emit the control constants with ODR linkage.
// This allows us to link against the symbols via '-mlink-builtin-bitcode'
// without preventing them from being optimized or causing symbol collisions.
[[gnu::alias("__oclc_unsafe_math_opt")]] const bool __oclc_unsafe_math_opt__;
[[gnu::alias("__oclc_daz_opt")]] const bool __oclc_daz_opt__;
[[gnu::alias("__oclc_correctly_rounded_sqrt32")]] const bool
    __oclc_correctly_rounded_sqrt32__;
[[gnu::alias("__oclc_finite_only_opt")]] const bool __oclc_finite_only_opt__;
[[gnu::alias("__oclc_wavefrontsize64")]] const bool __oclc_wavefrontsize64__;
[[gnu::alias("__oclc_ISA_version")]] const bool __oclc_ISA_version__;

#endif

#pragma omp end declare target

} // namespace platform

#endif
