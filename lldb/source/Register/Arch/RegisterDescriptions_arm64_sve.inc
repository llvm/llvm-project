//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

static uint32_t g_sve_s0_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z0,
    lldb_private::registers::arm64::id::fpu_v0,
    lldb_private::registers::arm64::id::fpu_d0};
static uint32_t g_sve_s1_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z1,
    lldb_private::registers::arm64::id::fpu_v1,
    lldb_private::registers::arm64::id::fpu_d1};
static uint32_t g_sve_s2_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z2,
    lldb_private::registers::arm64::id::fpu_v2,
    lldb_private::registers::arm64::id::fpu_d2};
static uint32_t g_sve_s3_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z3,
    lldb_private::registers::arm64::id::fpu_v3,
    lldb_private::registers::arm64::id::fpu_d3};
static uint32_t g_sve_s4_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z4,
    lldb_private::registers::arm64::id::fpu_v4,
    lldb_private::registers::arm64::id::fpu_d4};
static uint32_t g_sve_s5_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z5,
    lldb_private::registers::arm64::id::fpu_v5,
    lldb_private::registers::arm64::id::fpu_d5};
static uint32_t g_sve_s6_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z6,
    lldb_private::registers::arm64::id::fpu_v6,
    lldb_private::registers::arm64::id::fpu_d6};
static uint32_t g_sve_s7_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z7,
    lldb_private::registers::arm64::id::fpu_v7,
    lldb_private::registers::arm64::id::fpu_d7};
static uint32_t g_sve_s8_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z8,
    lldb_private::registers::arm64::id::fpu_v8,
    lldb_private::registers::arm64::id::fpu_d8};
static uint32_t g_sve_s9_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z9,
    lldb_private::registers::arm64::id::fpu_v9,
    lldb_private::registers::arm64::id::fpu_d9};
static uint32_t g_sve_s10_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z10,
    lldb_private::registers::arm64::id::fpu_v10,
    lldb_private::registers::arm64::id::fpu_d10};
static uint32_t g_sve_s11_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z11,
    lldb_private::registers::arm64::id::fpu_v11,
    lldb_private::registers::arm64::id::fpu_d11};
static uint32_t g_sve_s12_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z12,
    lldb_private::registers::arm64::id::fpu_v12,
    lldb_private::registers::arm64::id::fpu_d12};
static uint32_t g_sve_s13_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z13,
    lldb_private::registers::arm64::id::fpu_v13,
    lldb_private::registers::arm64::id::fpu_d13};
static uint32_t g_sve_s14_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z14,
    lldb_private::registers::arm64::id::fpu_v14,
    lldb_private::registers::arm64::id::fpu_d14};
static uint32_t g_sve_s15_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z15,
    lldb_private::registers::arm64::id::fpu_v15,
    lldb_private::registers::arm64::id::fpu_d15};
static uint32_t g_sve_s16_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z16,
    lldb_private::registers::arm64::id::fpu_v16,
    lldb_private::registers::arm64::id::fpu_d16};
static uint32_t g_sve_s17_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z17,
    lldb_private::registers::arm64::id::fpu_v17,
    lldb_private::registers::arm64::id::fpu_d17};
static uint32_t g_sve_s18_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z18,
    lldb_private::registers::arm64::id::fpu_v18,
    lldb_private::registers::arm64::id::fpu_d18};
static uint32_t g_sve_s19_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z19,
    lldb_private::registers::arm64::id::fpu_v19,
    lldb_private::registers::arm64::id::fpu_d19};
static uint32_t g_sve_s20_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z20,
    lldb_private::registers::arm64::id::fpu_v20,
    lldb_private::registers::arm64::id::fpu_d20};
static uint32_t g_sve_s21_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z21,
    lldb_private::registers::arm64::id::fpu_v21,
    lldb_private::registers::arm64::id::fpu_d21};
static uint32_t g_sve_s22_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z22,
    lldb_private::registers::arm64::id::fpu_v22,
    lldb_private::registers::arm64::id::fpu_d22};
static uint32_t g_sve_s23_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z23,
    lldb_private::registers::arm64::id::fpu_v23,
    lldb_private::registers::arm64::id::fpu_d23};
static uint32_t g_sve_s24_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z24,
    lldb_private::registers::arm64::id::fpu_v24,
    lldb_private::registers::arm64::id::fpu_d24};
static uint32_t g_sve_s25_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z25,
    lldb_private::registers::arm64::id::fpu_v25,
    lldb_private::registers::arm64::id::fpu_d25};
static uint32_t g_sve_s26_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z26,
    lldb_private::registers::arm64::id::fpu_v26,
    lldb_private::registers::arm64::id::fpu_d26};
static uint32_t g_sve_s27_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z27,
    lldb_private::registers::arm64::id::fpu_v27,
    lldb_private::registers::arm64::id::fpu_d27};
static uint32_t g_sve_s28_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z28,
    lldb_private::registers::arm64::id::fpu_v28,
    lldb_private::registers::arm64::id::fpu_d28};
static uint32_t g_sve_s29_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z29,
    lldb_private::registers::arm64::id::fpu_v29,
    lldb_private::registers::arm64::id::fpu_d29};
static uint32_t g_sve_s30_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z30,
    lldb_private::registers::arm64::id::fpu_v30,
    lldb_private::registers::arm64::id::fpu_d30};
static uint32_t g_sve_s31_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z31,
    lldb_private::registers::arm64::id::fpu_v31,
    lldb_private::registers::arm64::id::fpu_d31};

static uint32_t g_sve_d0_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z0,
    lldb_private::registers::arm64::id::fpu_v0,
    lldb_private::registers::arm64::id::fpu_s0};
static uint32_t g_sve_d1_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z1,
    lldb_private::registers::arm64::id::fpu_v1,
    lldb_private::registers::arm64::id::fpu_s1};
static uint32_t g_sve_d2_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z2,
    lldb_private::registers::arm64::id::fpu_v2,
    lldb_private::registers::arm64::id::fpu_s2};
static uint32_t g_sve_d3_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z3,
    lldb_private::registers::arm64::id::fpu_v3,
    lldb_private::registers::arm64::id::fpu_s3};
static uint32_t g_sve_d4_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z4,
    lldb_private::registers::arm64::id::fpu_v4,
    lldb_private::registers::arm64::id::fpu_s4};
static uint32_t g_sve_d5_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z5,
    lldb_private::registers::arm64::id::fpu_v5,
    lldb_private::registers::arm64::id::fpu_s5};
static uint32_t g_sve_d6_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z6,
    lldb_private::registers::arm64::id::fpu_v6,
    lldb_private::registers::arm64::id::fpu_s6};
static uint32_t g_sve_d7_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z7,
    lldb_private::registers::arm64::id::fpu_v7,
    lldb_private::registers::arm64::id::fpu_s7};
static uint32_t g_sve_d8_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z8,
    lldb_private::registers::arm64::id::fpu_v8,
    lldb_private::registers::arm64::id::fpu_s8};
static uint32_t g_sve_d9_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z9,
    lldb_private::registers::arm64::id::fpu_v9,
    lldb_private::registers::arm64::id::fpu_s9};
static uint32_t g_sve_d10_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z10,
    lldb_private::registers::arm64::id::fpu_v10,
    lldb_private::registers::arm64::id::fpu_s10};
static uint32_t g_sve_d11_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z11,
    lldb_private::registers::arm64::id::fpu_v11,
    lldb_private::registers::arm64::id::fpu_s11};
static uint32_t g_sve_d12_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z12,
    lldb_private::registers::arm64::id::fpu_v12,
    lldb_private::registers::arm64::id::fpu_s12};
static uint32_t g_sve_d13_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z13,
    lldb_private::registers::arm64::id::fpu_v13,
    lldb_private::registers::arm64::id::fpu_s13};
static uint32_t g_sve_d14_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z14,
    lldb_private::registers::arm64::id::fpu_v14,
    lldb_private::registers::arm64::id::fpu_s14};
static uint32_t g_sve_d15_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z15,
    lldb_private::registers::arm64::id::fpu_v15,
    lldb_private::registers::arm64::id::fpu_s15};
static uint32_t g_sve_d16_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z16,
    lldb_private::registers::arm64::id::fpu_v16,
    lldb_private::registers::arm64::id::fpu_s16};
static uint32_t g_sve_d17_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z17,
    lldb_private::registers::arm64::id::fpu_v17,
    lldb_private::registers::arm64::id::fpu_s17};
static uint32_t g_sve_d18_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z18,
    lldb_private::registers::arm64::id::fpu_v18,
    lldb_private::registers::arm64::id::fpu_s18};
static uint32_t g_sve_d19_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z19,
    lldb_private::registers::arm64::id::fpu_v19,
    lldb_private::registers::arm64::id::fpu_s19};
static uint32_t g_sve_d20_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z20,
    lldb_private::registers::arm64::id::fpu_v20,
    lldb_private::registers::arm64::id::fpu_s20};
static uint32_t g_sve_d21_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z21,
    lldb_private::registers::arm64::id::fpu_v21,
    lldb_private::registers::arm64::id::fpu_s21};
static uint32_t g_sve_d22_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z22,
    lldb_private::registers::arm64::id::fpu_v22,
    lldb_private::registers::arm64::id::fpu_s22};
static uint32_t g_sve_d23_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z23,
    lldb_private::registers::arm64::id::fpu_v23,
    lldb_private::registers::arm64::id::fpu_s23};
static uint32_t g_sve_d24_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z24,
    lldb_private::registers::arm64::id::fpu_v24,
    lldb_private::registers::arm64::id::fpu_s24};
static uint32_t g_sve_d25_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z25,
    lldb_private::registers::arm64::id::fpu_v25,
    lldb_private::registers::arm64::id::fpu_s25};
static uint32_t g_sve_d26_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z26,
    lldb_private::registers::arm64::id::fpu_v26,
    lldb_private::registers::arm64::id::fpu_s26};
static uint32_t g_sve_d27_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z27,
    lldb_private::registers::arm64::id::fpu_v27,
    lldb_private::registers::arm64::id::fpu_s27};
static uint32_t g_sve_d28_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z28,
    lldb_private::registers::arm64::id::fpu_v28,
    lldb_private::registers::arm64::id::fpu_s28};
static uint32_t g_sve_d29_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z29,
    lldb_private::registers::arm64::id::fpu_v29,
    lldb_private::registers::arm64::id::fpu_s29};
static uint32_t g_sve_d30_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z30,
    lldb_private::registers::arm64::id::fpu_v30,
    lldb_private::registers::arm64::id::fpu_s30};
static uint32_t g_sve_d31_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z31,
    lldb_private::registers::arm64::id::fpu_v31,
    lldb_private::registers::arm64::id::fpu_s31};

static uint32_t g_sve_v0_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z0,
    lldb_private::registers::arm64::id::fpu_d0,
    lldb_private::registers::arm64::id::fpu_s0};
static uint32_t g_sve_v1_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z1,
    lldb_private::registers::arm64::id::fpu_d1,
    lldb_private::registers::arm64::id::fpu_s1};
static uint32_t g_sve_v2_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z2,
    lldb_private::registers::arm64::id::fpu_d2,
    lldb_private::registers::arm64::id::fpu_s2};
static uint32_t g_sve_v3_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z3,
    lldb_private::registers::arm64::id::fpu_d3,
    lldb_private::registers::arm64::id::fpu_s3};
static uint32_t g_sve_v4_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z4,
    lldb_private::registers::arm64::id::fpu_d4,
    lldb_private::registers::arm64::id::fpu_s4};
static uint32_t g_sve_v5_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z5,
    lldb_private::registers::arm64::id::fpu_d5,
    lldb_private::registers::arm64::id::fpu_s5};
static uint32_t g_sve_v6_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z6,
    lldb_private::registers::arm64::id::fpu_d6,
    lldb_private::registers::arm64::id::fpu_s6};
static uint32_t g_sve_v7_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z7,
    lldb_private::registers::arm64::id::fpu_d7,
    lldb_private::registers::arm64::id::fpu_s7};
static uint32_t g_sve_v8_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z8,
    lldb_private::registers::arm64::id::fpu_d8,
    lldb_private::registers::arm64::id::fpu_s8};
static uint32_t g_sve_v9_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z9,
    lldb_private::registers::arm64::id::fpu_d9,
    lldb_private::registers::arm64::id::fpu_s9};
static uint32_t g_sve_v10_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z10,
    lldb_private::registers::arm64::id::fpu_d10,
    lldb_private::registers::arm64::id::fpu_s10};
static uint32_t g_sve_v11_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z11,
    lldb_private::registers::arm64::id::fpu_d11,
    lldb_private::registers::arm64::id::fpu_s11};
static uint32_t g_sve_v12_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z12,
    lldb_private::registers::arm64::id::fpu_d12,
    lldb_private::registers::arm64::id::fpu_s12};
static uint32_t g_sve_v13_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z13,
    lldb_private::registers::arm64::id::fpu_d13,
    lldb_private::registers::arm64::id::fpu_s13};
static uint32_t g_sve_v14_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z14,
    lldb_private::registers::arm64::id::fpu_d14,
    lldb_private::registers::arm64::id::fpu_s14};
static uint32_t g_sve_v15_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z15,
    lldb_private::registers::arm64::id::fpu_d15,
    lldb_private::registers::arm64::id::fpu_s15};
static uint32_t g_sve_v16_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z16,
    lldb_private::registers::arm64::id::fpu_d16,
    lldb_private::registers::arm64::id::fpu_s16};
static uint32_t g_sve_v17_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z17,
    lldb_private::registers::arm64::id::fpu_d17,
    lldb_private::registers::arm64::id::fpu_s17};
static uint32_t g_sve_v18_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z18,
    lldb_private::registers::arm64::id::fpu_d18,
    lldb_private::registers::arm64::id::fpu_s18};
static uint32_t g_sve_v19_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z19,
    lldb_private::registers::arm64::id::fpu_d19,
    lldb_private::registers::arm64::id::fpu_s19};
static uint32_t g_sve_v20_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z20,
    lldb_private::registers::arm64::id::fpu_d20,
    lldb_private::registers::arm64::id::fpu_s20};
static uint32_t g_sve_v21_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z21,
    lldb_private::registers::arm64::id::fpu_d21,
    lldb_private::registers::arm64::id::fpu_s21};
static uint32_t g_sve_v22_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z22,
    lldb_private::registers::arm64::id::fpu_d22,
    lldb_private::registers::arm64::id::fpu_s22};
static uint32_t g_sve_v23_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z23,
    lldb_private::registers::arm64::id::fpu_d23,
    lldb_private::registers::arm64::id::fpu_s23};
static uint32_t g_sve_v24_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z24,
    lldb_private::registers::arm64::id::fpu_d24,
    lldb_private::registers::arm64::id::fpu_s24};
static uint32_t g_sve_v25_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z25,
    lldb_private::registers::arm64::id::fpu_d25,
    lldb_private::registers::arm64::id::fpu_s25};
static uint32_t g_sve_v26_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z26,
    lldb_private::registers::arm64::id::fpu_d26,
    lldb_private::registers::arm64::id::fpu_s26};
static uint32_t g_sve_v27_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z27,
    lldb_private::registers::arm64::id::fpu_d27,
    lldb_private::registers::arm64::id::fpu_s27};
static uint32_t g_sve_v28_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z28,
    lldb_private::registers::arm64::id::fpu_d28,
    lldb_private::registers::arm64::id::fpu_s28};
static uint32_t g_sve_v29_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z29,
    lldb_private::registers::arm64::id::fpu_d29,
    lldb_private::registers::arm64::id::fpu_s29};
static uint32_t g_sve_v30_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z30,
    lldb_private::registers::arm64::id::fpu_d30,
    lldb_private::registers::arm64::id::fpu_s30};
static uint32_t g_sve_v31_invalidates[] = {
    lldb_private::registers::arm64::id::sve_z31,
    lldb_private::registers::arm64::id::fpu_d31,
    lldb_private::registers::arm64::id::fpu_s31};

static uint32_t g_contained_z0[] = {lldb_private::registers::arm64::id::sve_z0};
static uint32_t g_contained_z1[] = {lldb_private::registers::arm64::id::sve_z1};
static uint32_t g_contained_z2[] = {lldb_private::registers::arm64::id::sve_z2};
static uint32_t g_contained_z3[] = {lldb_private::registers::arm64::id::sve_z3};
static uint32_t g_contained_z4[] = {lldb_private::registers::arm64::id::sve_z4};
static uint32_t g_contained_z5[] = {lldb_private::registers::arm64::id::sve_z5};
static uint32_t g_contained_z6[] = {lldb_private::registers::arm64::id::sve_z6};
static uint32_t g_contained_z7[] = {lldb_private::registers::arm64::id::sve_z7};
static uint32_t g_contained_z8[] = {lldb_private::registers::arm64::id::sve_z8};
static uint32_t g_contained_z9[] = {lldb_private::registers::arm64::id::sve_z9};
static uint32_t g_contained_z10[] = {
    lldb_private::registers::arm64::id::sve_z10};
static uint32_t g_contained_z11[] = {
    lldb_private::registers::arm64::id::sve_z11};
static uint32_t g_contained_z12[] = {
    lldb_private::registers::arm64::id::sve_z12};
static uint32_t g_contained_z13[] = {
    lldb_private::registers::arm64::id::sve_z13};
static uint32_t g_contained_z14[] = {
    lldb_private::registers::arm64::id::sve_z14};
static uint32_t g_contained_z15[] = {
    lldb_private::registers::arm64::id::sve_z15};
static uint32_t g_contained_z16[] = {
    lldb_private::registers::arm64::id::sve_z16};
static uint32_t g_contained_z17[] = {
    lldb_private::registers::arm64::id::sve_z17};
static uint32_t g_contained_z18[] = {
    lldb_private::registers::arm64::id::sve_z18};
static uint32_t g_contained_z19[] = {
    lldb_private::registers::arm64::id::sve_z19};
static uint32_t g_contained_z20[] = {
    lldb_private::registers::arm64::id::sve_z20};
static uint32_t g_contained_z21[] = {
    lldb_private::registers::arm64::id::sve_z21};
static uint32_t g_contained_z22[] = {
    lldb_private::registers::arm64::id::sve_z22};
static uint32_t g_contained_z23[] = {
    lldb_private::registers::arm64::id::sve_z23};
static uint32_t g_contained_z24[] = {
    lldb_private::registers::arm64::id::sve_z24};
static uint32_t g_contained_z25[] = {
    lldb_private::registers::arm64::id::sve_z25};
static uint32_t g_contained_z26[] = {
    lldb_private::registers::arm64::id::sve_z26};
static uint32_t g_contained_z27[] = {
    lldb_private::registers::arm64::id::sve_z27};
static uint32_t g_contained_z28[] = {
    lldb_private::registers::arm64::id::sve_z28};
static uint32_t g_contained_z29[] = {
    lldb_private::registers::arm64::id::sve_z29};
static uint32_t g_contained_z30[] = {
    lldb_private::registers::arm64::id::sve_z30};
static uint32_t g_contained_z31[] = {
    lldb_private::registers::arm64::id::sve_z31};

#define SVE_REG_KIND(reg) MISC_KIND(reg, sve, LLDB_INVALID_REGNUM)
#define MISC_VG_KIND(lldb_kind) MISC_KIND(vg, sve, LLDB_INVALID_REGNUM)

// Default offset SVE Z registers and all corresponding pseudo registers
// ( S, D and V registers) is zero and will be configured during execution.

// Defines sve pseudo vector (V) register with 16-byte size
#define DEFINE_VREG_SVE(vreg, zreg)                                            \
  RegisterDescriptor(lldb_private::registers::arm64::id::fpu##_##vreg, #vreg, nullptr, 16,                 \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     VREG_KIND(vreg), g_contained_##zreg,                      \
                     g_sve_##vreg##_invalidates, nullptr)

// Defines S and D pseudo registers mapping over corresponding vector register
#define DEFINE_FPU_PSEUDO_SVE(reg, size, zreg)                                 \
  RegisterDescriptor(lldb_private::registers::arm64::id::fpu_##reg, #reg, nullptr, size,                   \
                     lldb::eEncodingIEEE754, lldb::eFormatFloat,               \
                     LLDB_KIND(lldb_private::registers::arm64::id::fpu_##reg), g_contained_##zreg,         \
                     g_sve_##reg##_invalidates, nullptr)

// Defines a Z vector register with 16-byte default size
#define DEFINE_ZREG(reg)                                                       \
  RegisterDescriptor(lldb_private::registers::arm64::id::sve##_##reg, #reg, nullptr, 16,                   \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     SVE_REG_KIND(reg), {}, {}, nullptr)

// Defines a P vector register with 2-byte default size
#define DEFINE_PREG(reg)                                                       \
  RegisterDescriptor(lldb_private::registers::arm64::id::sve##_##reg, #reg, nullptr, 2,                    \
                     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,        \
                     SVE_REG_KIND(reg), {}, {}, nullptr)

static const RegisterDescriptor g_register_infos_arm64_sve_le[] = {
    // DEFINE_GPR64(name, GENERIC KIND)
    DEFINE_GPR64(x0, LLDB_REGNUM_GENERIC_ARG1),
    DEFINE_GPR64(x1, LLDB_REGNUM_GENERIC_ARG2),
    DEFINE_GPR64(x2, LLDB_REGNUM_GENERIC_ARG3),
    DEFINE_GPR64(x3, LLDB_REGNUM_GENERIC_ARG4),
    DEFINE_GPR64(x4, LLDB_REGNUM_GENERIC_ARG5),
    DEFINE_GPR64(x5, LLDB_REGNUM_GENERIC_ARG6),
    DEFINE_GPR64(x6, LLDB_REGNUM_GENERIC_ARG7),
    DEFINE_GPR64(x7, LLDB_REGNUM_GENERIC_ARG8),
    DEFINE_GPR64(x8, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x9, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x10, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x11, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x12, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x13, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x14, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x15, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x16, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x17, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x18, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x19, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x20, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x21, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x22, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x23, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x24, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x25, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x26, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x27, LLDB_INVALID_REGNUM),
    DEFINE_GPR64(x28, LLDB_INVALID_REGNUM),
    // DEFINE_GPR64(name, GENERIC KIND)
    DEFINE_GPR64_ALT(fp, x29, LLDB_REGNUM_GENERIC_FP),
    DEFINE_GPR64_ALT(lr, x30, LLDB_REGNUM_GENERIC_RA),
    DEFINE_GPR64_ALT(sp, x31, LLDB_REGNUM_GENERIC_SP),
    DEFINE_GPR64(pc, LLDB_REGNUM_GENERIC_PC),

    DEFINE_GPR_CPSR,

    // DEFINE_GPR32(name, parent name)
    DEFINE_GPR32(w0, x0), DEFINE_GPR32(w1, x1), DEFINE_GPR32(w2, x2),
    DEFINE_GPR32(w3, x3), DEFINE_GPR32(w4, x4), DEFINE_GPR32(w5, x5),
    DEFINE_GPR32(w6, x6), DEFINE_GPR32(w7, x7), DEFINE_GPR32(w8, x8),
    DEFINE_GPR32(w9, x9), DEFINE_GPR32(w10, x10), DEFINE_GPR32(w11, x11),
    DEFINE_GPR32(w12, x12), DEFINE_GPR32(w13, x13), DEFINE_GPR32(w14, x14),
    DEFINE_GPR32(w15, x15), DEFINE_GPR32(w16, x16), DEFINE_GPR32(w17, x17),
    DEFINE_GPR32(w18, x18), DEFINE_GPR32(w19, x19), DEFINE_GPR32(w20, x20),
    DEFINE_GPR32(w21, x21), DEFINE_GPR32(w22, x22), DEFINE_GPR32(w23, x23),
    DEFINE_GPR32(w24, x24), DEFINE_GPR32(w25, x25), DEFINE_GPR32(w26, x26),
    DEFINE_GPR32(w27, x27), DEFINE_GPR32(w28, x28),

    // DEFINE_VREG_SVE(v register, z register)
    DEFINE_VREG_SVE(v0, z0), DEFINE_VREG_SVE(v1, z1), DEFINE_VREG_SVE(v2, z2),
    DEFINE_VREG_SVE(v3, z3), DEFINE_VREG_SVE(v4, z4), DEFINE_VREG_SVE(v5, z5),
    DEFINE_VREG_SVE(v6, z6), DEFINE_VREG_SVE(v7, z7), DEFINE_VREG_SVE(v8, z8),
    DEFINE_VREG_SVE(v9, z9), DEFINE_VREG_SVE(v10, z10),
    DEFINE_VREG_SVE(v11, z11), DEFINE_VREG_SVE(v12, z12),
    DEFINE_VREG_SVE(v13, z13), DEFINE_VREG_SVE(v14, z14),
    DEFINE_VREG_SVE(v15, z15), DEFINE_VREG_SVE(v16, z16),
    DEFINE_VREG_SVE(v17, z17), DEFINE_VREG_SVE(v18, z18),
    DEFINE_VREG_SVE(v19, z19), DEFINE_VREG_SVE(v20, z20),
    DEFINE_VREG_SVE(v21, z21), DEFINE_VREG_SVE(v22, z22),
    DEFINE_VREG_SVE(v23, z23), DEFINE_VREG_SVE(v24, z24),
    DEFINE_VREG_SVE(v25, z25), DEFINE_VREG_SVE(v26, z26),
    DEFINE_VREG_SVE(v27, z27), DEFINE_VREG_SVE(v28, z28),
    DEFINE_VREG_SVE(v29, z29), DEFINE_VREG_SVE(v30, z30),
    DEFINE_VREG_SVE(v31, z31),

    // DEFINE_FPU_PSEUDO(name, size, parent register)
    DEFINE_FPU_PSEUDO_SVE(s0, 4, z0), DEFINE_FPU_PSEUDO_SVE(s1, 4, z1),
    DEFINE_FPU_PSEUDO_SVE(s2, 4, z2), DEFINE_FPU_PSEUDO_SVE(s3, 4, z3),
    DEFINE_FPU_PSEUDO_SVE(s4, 4, z4), DEFINE_FPU_PSEUDO_SVE(s5, 4, z5),
    DEFINE_FPU_PSEUDO_SVE(s6, 4, z6), DEFINE_FPU_PSEUDO_SVE(s7, 4, z7),
    DEFINE_FPU_PSEUDO_SVE(s8, 4, z8), DEFINE_FPU_PSEUDO_SVE(s9, 4, z9),
    DEFINE_FPU_PSEUDO_SVE(s10, 4, z10), DEFINE_FPU_PSEUDO_SVE(s11, 4, z11),
    DEFINE_FPU_PSEUDO_SVE(s12, 4, z12), DEFINE_FPU_PSEUDO_SVE(s13, 4, z13),
    DEFINE_FPU_PSEUDO_SVE(s14, 4, z14), DEFINE_FPU_PSEUDO_SVE(s15, 4, z15),
    DEFINE_FPU_PSEUDO_SVE(s16, 4, z16), DEFINE_FPU_PSEUDO_SVE(s17, 4, z17),
    DEFINE_FPU_PSEUDO_SVE(s18, 4, z18), DEFINE_FPU_PSEUDO_SVE(s19, 4, z19),
    DEFINE_FPU_PSEUDO_SVE(s20, 4, z20), DEFINE_FPU_PSEUDO_SVE(s21, 4, z21),
    DEFINE_FPU_PSEUDO_SVE(s22, 4, z22), DEFINE_FPU_PSEUDO_SVE(s23, 4, z23),
    DEFINE_FPU_PSEUDO_SVE(s24, 4, z24), DEFINE_FPU_PSEUDO_SVE(s25, 4, z25),
    DEFINE_FPU_PSEUDO_SVE(s26, 4, z26), DEFINE_FPU_PSEUDO_SVE(s27, 4, z27),
    DEFINE_FPU_PSEUDO_SVE(s28, 4, z28), DEFINE_FPU_PSEUDO_SVE(s29, 4, z29),
    DEFINE_FPU_PSEUDO_SVE(s30, 4, z30), DEFINE_FPU_PSEUDO_SVE(s31, 4, z31),

    DEFINE_FPU_PSEUDO_SVE(d0, 8, z0), DEFINE_FPU_PSEUDO_SVE(d1, 8, z1),
    DEFINE_FPU_PSEUDO_SVE(d2, 8, z2), DEFINE_FPU_PSEUDO_SVE(d3, 8, z3),
    DEFINE_FPU_PSEUDO_SVE(d4, 8, z4), DEFINE_FPU_PSEUDO_SVE(d5, 8, z5),
    DEFINE_FPU_PSEUDO_SVE(d6, 8, z6), DEFINE_FPU_PSEUDO_SVE(d7, 8, z7),
    DEFINE_FPU_PSEUDO_SVE(d8, 8, z8), DEFINE_FPU_PSEUDO_SVE(d9, 8, z9),
    DEFINE_FPU_PSEUDO_SVE(d10, 8, z10), DEFINE_FPU_PSEUDO_SVE(d11, 8, z11),
    DEFINE_FPU_PSEUDO_SVE(d12, 8, z12), DEFINE_FPU_PSEUDO_SVE(d13, 8, z13),
    DEFINE_FPU_PSEUDO_SVE(d14, 8, z14), DEFINE_FPU_PSEUDO_SVE(d15, 8, z15),
    DEFINE_FPU_PSEUDO_SVE(d16, 8, z16), DEFINE_FPU_PSEUDO_SVE(d17, 8, z17),
    DEFINE_FPU_PSEUDO_SVE(d18, 8, z18), DEFINE_FPU_PSEUDO_SVE(d19, 8, z19),
    DEFINE_FPU_PSEUDO_SVE(d20, 8, z20), DEFINE_FPU_PSEUDO_SVE(d21, 8, z21),
    DEFINE_FPU_PSEUDO_SVE(d22, 8, z22), DEFINE_FPU_PSEUDO_SVE(d23, 8, z23),
    DEFINE_FPU_PSEUDO_SVE(d24, 8, z24), DEFINE_FPU_PSEUDO_SVE(d25, 8, z25),
    DEFINE_FPU_PSEUDO_SVE(d26, 8, z26), DEFINE_FPU_PSEUDO_SVE(d27, 8, z27),
    DEFINE_FPU_PSEUDO_SVE(d28, 8, z28), DEFINE_FPU_PSEUDO_SVE(d29, 8, z29),
    DEFINE_FPU_PSEUDO_SVE(d30, 8, z30), DEFINE_FPU_PSEUDO_SVE(d31, 8, z31),

    // DEFINE_MISC_LLDB_REGS(name, size, TYPE, lldb kind)
    DEFINE_MISC_LLDB_REGS(fpsr, 4, FPU,
                          lldb_private::registers::arm64::id::fpu_fpsr),
    DEFINE_MISC_LLDB_REGS(fpcr, 4, FPU,
                          lldb_private::registers::arm64::id::fpu_fpcr),

    DEFINE_MISC_REGS(vg, 8, VG, lldb_private::registers::arm64::id::sve_vg),
    // DEFINE_ZREG(name)
    DEFINE_ZREG(z0), DEFINE_ZREG(z1), DEFINE_ZREG(z2), DEFINE_ZREG(z3),
    DEFINE_ZREG(z4), DEFINE_ZREG(z5), DEFINE_ZREG(z6), DEFINE_ZREG(z7),
    DEFINE_ZREG(z8), DEFINE_ZREG(z9), DEFINE_ZREG(z10), DEFINE_ZREG(z11),
    DEFINE_ZREG(z12), DEFINE_ZREG(z13), DEFINE_ZREG(z14), DEFINE_ZREG(z15),
    DEFINE_ZREG(z16), DEFINE_ZREG(z17), DEFINE_ZREG(z18), DEFINE_ZREG(z19),
    DEFINE_ZREG(z20), DEFINE_ZREG(z21), DEFINE_ZREG(z22), DEFINE_ZREG(z23),
    DEFINE_ZREG(z24), DEFINE_ZREG(z25), DEFINE_ZREG(z26), DEFINE_ZREG(z27),
    DEFINE_ZREG(z28), DEFINE_ZREG(z29), DEFINE_ZREG(z30), DEFINE_ZREG(z31),

    // DEFINE_PREG(name)
    DEFINE_PREG(p0), DEFINE_PREG(p1), DEFINE_PREG(p2), DEFINE_PREG(p3),
    DEFINE_PREG(p4), DEFINE_PREG(p5), DEFINE_PREG(p6), DEFINE_PREG(p7),
    DEFINE_PREG(p8), DEFINE_PREG(p9), DEFINE_PREG(p10), DEFINE_PREG(p11),
    DEFINE_PREG(p12), DEFINE_PREG(p13), DEFINE_PREG(p14), DEFINE_PREG(p15),

    // DEFINE FFR
    DEFINE_PREG(ffr)};
