; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

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
; CHECK-DAG: .b8 GblU[12] = {7, 6, 0, 0, 5, 4, 3, 2, 1, 0, 0, 0};
@GblU = global {i16, i32, i8} {i16 1543, i32 33752069, i8 1}

