; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -use-constant-int-for-fixed-length-splat | FileCheck %s

; Make sure the globals constant initializers are not prone to host endianess 
; issues.

; CHECK-DAG: .b8 Gbli08[2] = {171, 205};
@Gbli08 = global [2 x i8] [i8 171, i8 205]

; CHECK-DAG: .b8 Gbli16[4] = {205, 171, 1, 239};
@Gbli16 = global [2 x i16] [i16 43981, i16 61185]

; CHECK-DAG: .b8 Gbli32[8] = {1, 239, 205, 171, 137, 103, 69, 35};
@Gbli32 = global [2 x i32] [i32 2882400001, i32 591751049]

; CHECK-DAG: .b8 Gbli64[16] = {137, 103, 69, 35, 1, 239, 205, 171, 239, 205, 171, 137, 103, 69, 35, 1};
@Gbli64 = global [2 x i64] [i64 12379813738877118345, i64 81985529216486895]

; CHECK-DAG: .b8 Gbli128[32] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
@Gbli128 = global [2 x i128] [i128  1339673755198158349044581307228491536, i128 21345817372864405881847059188222722561]

; CHECK-DAG: .b8 Gblf32[8] = {192, 225, 100, 75, 0, 96, 106, 69};
@Gblf32 = global [2 x float] [float 1.5e+7, float 3.75e+3]

; CHECK-DAG: .b8 Gblf64[16] = {116, 10, 181, 48, 134, 62, 230, 58, 106, 222, 138, 98, 204, 250, 200, 75};
@Gblf64 = global [2 x double] [double 5.75e-25, double 12.25e+56]

; Make sure we fill in alignment gaps correctly.
; CHECK-DAG: .b8 GblU[12] = {7, 6, 0, 0, 5, 4, 3, 2, 1};
@GblU = global {i16, i32, i8} {i16 1543, i32 33752069, i8 1}

; CHECK-DAG: .b8 Gblv4i16[8] = {205, 171, 205, 171, 205, 171, 205, 171};
@Gblv4i16 = global <4 x i16> splat(i16 43981)

; CHECK-DAG: .b8 Gblv4f32[16] = {0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63};
@Gblv4f32 = global <4 x float> splat(float 1.0)

; Large-integer globals whose bit width is not a multiple of 8 must emit
; their full ceil(bitWidth/8) bytes, including the top partial byte.
; CHECK-DAG: .b8 globalint_Gbli65[9] = {255, 255, 255, 255, 255, 255, 255, 255, 1};
@globalint_Gbli65 = global i65 u0x1FFFFFFFFFFFFFFFF

; CHECK-DAG: .b8 globalint_Gbli100[13] = {186, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15};
@globalint_Gbli100 = global i100 u0xF000000000000000000000ABA

; Non-power-of-2 vector (<3 x i32> store=12 alloc=16) in the middle of the
; struct must keep its tail padding.
; CHECK-DAG: .b8 nonpow2vec_field_pad[32] = {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 254, 255, 255, 255};
%struct.nonpow2vec_pad = type { <3 x i32>, i32 }
@nonpow2vec_field_pad = global %struct.nonpow2vec_pad { <3 x i32> <i32 1, i32 2, i32 3>, i32 -2 }

; CHECK-DAG: .b8 nonpow2vec_arr_pad[32] = {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6};
@nonpow2vec_arr_pad = global [2 x <3 x i32>] [<3 x i32> <i32 1, i32 2, i32 3>, <3 x i32> <i32 4, i32 5, i32 6>]

; Vector splats nested in an array or a struct must be emitted, and keep their
; tail padding (<3 x float> store=12 alloc=16, <3 x i16> store=6 alloc=8).
; CHECK-DAG: .b8 splat_arr[32] = {0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64};
@splat_arr = global [2 x <4 x float>] [<4 x float> splat(float 1.0), <4 x float> splat(float 2.0)]

; CHECK-DAG: .b8 splat_field_pad[32] = {0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 0, 0, 254, 255, 255, 255};
@splat_field_pad = global {<3 x float>, i32} {<3 x float> splat(float 1.0), i32 -2}

; CHECK-DAG: .b8 splat_arr_pad[16] = {205, 171, 205, 171, 205, 171, 0, 0, 1, 1, 1, 1, 1, 1};
@splat_arr_pad = global [2 x <3 x i16>] [<3 x i16> splat(i16 43981), <3 x i16> splat(i16 257)]

; CHECK-DAG: .b8 splat_half_arr[8] = {0, 60, 0, 60, 0, 64, 0, 64};
@splat_half_arr = global [2 x <2 x half>] [<2 x half> splat(half 1.0), <2 x half> splat(half 2.0)]

; CHECK-DAG: .b8 splat_bfloat_arr[8] = {128, 63, 128, 63, 0, 64, 0, 64};
@splat_bfloat_arr = global [2 x <2 x bfloat>] [<2 x bfloat> splat(bfloat 1.0), <2 x bfloat> splat(bfloat 2.0)]

; CHECK-DAG: .b8 splat_struct_arr[32] = {0, 0, 128, 63, 0, 0, 128, 63, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 64, 4};
@splat_struct_arr = global [2 x {<2 x float>, i32}] [{<2 x float>, i32} {<2 x float> splat(float 1.0), i32 3}, {<2 x float>, i32} {<2 x float> splat(float 2.0), i32 4}]

; Elements that aren't a whole number of bytes are bit-packed, whatever the
; enclosing aggregate and whether or not the vector is a splat.
; CHECK-DAG: .b8 packed_splat[2] = {17, 17};
@packed_splat = global <4 x i4> splat(i4 1)

; CHECK-DAG: .b8 packed_splat_arr[4] = {17, 17, 34, 34};
@packed_splat_arr = global [2 x <4 x i4>] [<4 x i4> splat(i4 1), <4 x i4> splat(i4 2)]

; CHECK-DAG: .b8 packed_splat_field[2] = {17, 7};
@packed_splat_field = global {<2 x i4>, i8} {<2 x i4> splat(i4 1), i8 7}

; CHECK-DAG: .b8 packed_arr[4] = {33, 67, 101, 135};
@packed_arr = global [4 x <2 x i4>] [<2 x i4> <i4 1, i4 2>, <2 x i4> <i4 3, i4 4>, <2 x i4> <i4 5, i4 6>, <2 x i4> <i4 7, i4 8>]

; CHECK-DAG: .b8 packed_i12[3] = {1, 32};
@packed_i12 = global <2 x i12> <i12 1, i12 2>
