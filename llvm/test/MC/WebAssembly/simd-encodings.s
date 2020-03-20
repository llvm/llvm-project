# RUN: llvm-mc -show-encoding -triple=wasm32-unkown-unknown -mattr=+unimplemented-simd128 < %s | FileCheck %s

main:
    .functype main () -> ()

    # CHECK: v128.load 48 # encoding: [0xfd,0x00,0x04,0x30]
    v128.load 48

    # CHECK: v128.store 48 # encoding: [0xfd,0x01,0x04,0x30]
    v128.store 48

    # CHECK: v128.const 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    # CHECK-SAME: # encoding: [0xfd,0x02,
    # CHECK-SAME: 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    # CHECK-SAME: 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f]
    v128.const 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

    # CHECK: v128.const 256, 770, 1284, 1798, 2312, 2826, 3340, 3854
    # CHECK-SAME: # encoding: [0xfd,0x02,
    # CHECK-SAME: 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    # CHECK-SAME: 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f]
    v128.const 256, 770, 1284, 1798, 2312, 2826, 3340, 3854

    # TODO(tlively): Fix assembler so v128.const works with 4xi32 and 2xi64

    # CHECK: v128.const 0x1.0402p-121, 0x1.0c0a08p-113,
    # CHECK-SAME:       0x1.14121p-105, 0x1.1c1a18p-97
    # CHECK-SAME: # encoding: [0xfd,0x02,
    # CHECK-SAME: 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    # CHECK-SAME: 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f]
    v128.const 0x1.0402p-121, 0x1.0c0a08p-113, 0x1.14121p-105, 0x1.1c1a18p-97

    # CHECK: v128.const 0x1.60504030201p-911, 0x1.e0d0c0b0a0908p-783
    # CHECK-SAME: # encoding: [0xfd,0x02,
    # CHECK-SAME: 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    # CHECK-SAME: 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f]
    v128.const 0x1.60504030201p-911, 0x1.e0d0c0b0a0908p-783

    # CHECK: v8x16.shuffle 0, 17, 2, 19, 4, 21, 6, 23,
    # CHECK-SAME:          8, 25, 10, 27, 12, 29, 14, 31
    # CHECK-SAME: # encoding: [0xfd,0x03,
    # CHECK-SAME: 0x00,0x11,0x02,0x13,0x04,0x15,0x06,0x17,
    # CHECK-SAME: 0x08,0x19,0x0a,0x1b,0x0c,0x1d,0x0e,0x1f]
    v8x16.shuffle 0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31

    # CHECK: i8x16.splat # encoding: [0xfd,0x04]
    i8x16.splat

    # CHECK: i8x16.extract_lane_s 15 # encoding: [0xfd,0x05,0x0f]
    i8x16.extract_lane_s 15

    # CHECK: i8x16.extract_lane_u 15 # encoding: [0xfd,0x06,0x0f]
    i8x16.extract_lane_u 15

    # CHECK: i8x16.replace_lane 15 # encoding: [0xfd,0x07,0x0f]
    i8x16.replace_lane 15

    # CHECK: i16x8.splat # encoding: [0xfd,0x08]
    i16x8.splat

    # CHECK: i16x8.extract_lane_s 7 # encoding: [0xfd,0x09,0x07]
    i16x8.extract_lane_s 7

    # CHECK: i16x8.extract_lane_u 7 # encoding: [0xfd,0x0a,0x07]
    i16x8.extract_lane_u 7

    # CHECK: i16x8.replace_lane 7 # encoding: [0xfd,0x0b,0x07]
    i16x8.replace_lane 7

    # CHECK: i32x4.splat # encoding: [0xfd,0x0c]
    i32x4.splat

    # CHECK: i32x4.extract_lane 3 # encoding: [0xfd,0x0d,0x03]
    i32x4.extract_lane 3

    # CHECK: i32x4.replace_lane 3 # encoding: [0xfd,0x0e,0x03]
    i32x4.replace_lane 3

    # CHECK: i64x2.splat # encoding: [0xfd,0x0f]
    i64x2.splat

    # CHECK: i64x2.extract_lane 1 # encoding: [0xfd,0x10,0x01]
    i64x2.extract_lane 1

    # CHECK: i64x2.replace_lane 1 # encoding: [0xfd,0x11,0x01]
    i64x2.replace_lane 1

    # CHECK: f32x4.splat # encoding: [0xfd,0x12]
    f32x4.splat

    # CHECK: f32x4.extract_lane 3 # encoding: [0xfd,0x13,0x03]
    f32x4.extract_lane 3

    # CHECK: f32x4.replace_lane 3 # encoding: [0xfd,0x14,0x03]
    f32x4.replace_lane 3

    # CHECK: f64x2.splat # encoding: [0xfd,0x15]
    f64x2.splat

    # CHECK: f64x2.extract_lane 1 # encoding: [0xfd,0x16,0x01]
    f64x2.extract_lane 1

    # CHECK: f64x2.replace_lane 1 # encoding: [0xfd,0x17,0x01]
    f64x2.replace_lane 1

    # CHECK: i8x16.eq # encoding: [0xfd,0x18]
    i8x16.eq

    # CHECK: i8x16.ne # encoding: [0xfd,0x19]
    i8x16.ne

    # CHECK: i8x16.lt_s # encoding: [0xfd,0x1a]
    i8x16.lt_s

    # CHECK: i8x16.lt_u # encoding: [0xfd,0x1b]
    i8x16.lt_u

    # CHECK: i8x16.gt_s # encoding: [0xfd,0x1c]
    i8x16.gt_s

    # CHECK: i8x16.gt_u # encoding: [0xfd,0x1d]
    i8x16.gt_u

    # CHECK: i8x16.le_s # encoding: [0xfd,0x1e]
    i8x16.le_s

    # CHECK: i8x16.le_u # encoding: [0xfd,0x1f]
    i8x16.le_u

    # CHECK: i8x16.ge_s # encoding: [0xfd,0x20]
    i8x16.ge_s

    # CHECK: i8x16.ge_u # encoding: [0xfd,0x21]
    i8x16.ge_u

    # CHECK: i16x8.eq # encoding: [0xfd,0x22]
    i16x8.eq

    # CHECK: i16x8.ne # encoding: [0xfd,0x23]
    i16x8.ne

    # CHECK: i16x8.lt_s # encoding: [0xfd,0x24]
    i16x8.lt_s

    # CHECK: i16x8.lt_u # encoding: [0xfd,0x25]
    i16x8.lt_u

    # CHECK: i16x8.gt_s # encoding: [0xfd,0x26]
    i16x8.gt_s

    # CHECK: i16x8.gt_u # encoding: [0xfd,0x27]
    i16x8.gt_u

    # CHECK: i16x8.le_s # encoding: [0xfd,0x28]
    i16x8.le_s

    # CHECK: i16x8.le_u # encoding: [0xfd,0x29]
    i16x8.le_u

    # CHECK: i16x8.ge_s # encoding: [0xfd,0x2a]
    i16x8.ge_s

    # CHECK: i16x8.ge_u # encoding: [0xfd,0x2b]
    i16x8.ge_u

    # CHECK: i32x4.eq # encoding: [0xfd,0x2c]
    i32x4.eq

    # CHECK: i32x4.ne # encoding: [0xfd,0x2d]
    i32x4.ne

    # CHECK: i32x4.lt_s # encoding: [0xfd,0x2e]
    i32x4.lt_s

    # CHECK: i32x4.lt_u # encoding: [0xfd,0x2f]
    i32x4.lt_u

    # CHECK: i32x4.gt_s # encoding: [0xfd,0x30]
    i32x4.gt_s

    # CHECK: i32x4.gt_u # encoding: [0xfd,0x31]
    i32x4.gt_u

    # CHECK: i32x4.le_s # encoding: [0xfd,0x32]
    i32x4.le_s

    # CHECK: i32x4.le_u # encoding: [0xfd,0x33]
    i32x4.le_u

    # CHECK: i32x4.ge_s # encoding: [0xfd,0x34]
    i32x4.ge_s

    # CHECK: i32x4.ge_u # encoding: [0xfd,0x35]
    i32x4.ge_u

    # CHECK: f32x4.eq # encoding: [0xfd,0x40]
    f32x4.eq

    # CHECK: f32x4.ne # encoding: [0xfd,0x41]
    f32x4.ne

    # CHECK: f32x4.lt # encoding: [0xfd,0x42]
    f32x4.lt

    # CHECK: f32x4.gt # encoding: [0xfd,0x43]
    f32x4.gt

    # CHECK: f32x4.le # encoding: [0xfd,0x44]
    f32x4.le

    # CHECK: f32x4.ge # encoding: [0xfd,0x45]
    f32x4.ge

    # CHECK: f64x2.eq # encoding: [0xfd,0x46]
    f64x2.eq

    # CHECK: f64x2.ne # encoding: [0xfd,0x47]
    f64x2.ne

    # CHECK: f64x2.lt # encoding: [0xfd,0x48]
    f64x2.lt

    # CHECK: f64x2.gt # encoding: [0xfd,0x49]
    f64x2.gt

    # CHECK: f64x2.le # encoding: [0xfd,0x4a]
    f64x2.le

    # CHECK: f64x2.ge # encoding: [0xfd,0x4b]
    f64x2.ge

    # CHECK: v128.not # encoding: [0xfd,0x4c]
    v128.not

    # CHECK: v128.and # encoding: [0xfd,0x4d]
    v128.and

    # CHECK: v128.or # encoding: [0xfd,0x4e]
    v128.or

    # CHECK: v128.xor # encoding: [0xfd,0x4f]
    v128.xor

    # CHECK: v128.bitselect # encoding: [0xfd,0x50]
    v128.bitselect

    # CHECK: i8x16.neg # encoding: [0xfd,0x51]
    i8x16.neg

    # CHECK: i8x16.any_true # encoding: [0xfd,0x52]
    i8x16.any_true

    # CHECK: i8x16.all_true # encoding: [0xfd,0x53]
    i8x16.all_true

    # CHECK: i8x16.shl # encoding: [0xfd,0x54]
    i8x16.shl

    # CHECK: i8x16.shr_s # encoding: [0xfd,0x55]
    i8x16.shr_s

    # CHECK: i8x16.shr_u # encoding: [0xfd,0x56]
    i8x16.shr_u

    # CHECK: i8x16.add # encoding: [0xfd,0x57]
    i8x16.add

    # CHECK: i8x16.add_saturate_s # encoding: [0xfd,0x58]
    i8x16.add_saturate_s

    # CHECK: i8x16.add_saturate_u # encoding: [0xfd,0x59]
    i8x16.add_saturate_u

    # CHECK: i8x16.sub # encoding: [0xfd,0x5a]
    i8x16.sub

    # CHECK: i8x16.sub_saturate_s # encoding: [0xfd,0x5b]
    i8x16.sub_saturate_s

    # CHECK: i8x16.sub_saturate_u # encoding: [0xfd,0x5c]
    i8x16.sub_saturate_u

    # CHECK: i8x16.mul # encoding: [0xfd,0x5d]
    i8x16.mul

    # CHECK: i8x16.min_s # encoding: [0xfd,0x5e]
    i8x16.min_s

    # CHECK: i8x16.min_u # encoding: [0xfd,0x5f]
    i8x16.min_u

    # CHECK: i8x16.max_s # encoding: [0xfd,0x60]
    i8x16.max_s

    # CHECK: i8x16.max_u # encoding: [0xfd,0x61]
    i8x16.max_u

    # CHECK: i16x8.neg # encoding: [0xfd,0x62]
    i16x8.neg

    # CHECK: i16x8.any_true # encoding: [0xfd,0x63]
    i16x8.any_true

    # CHECK: i16x8.all_true # encoding: [0xfd,0x64]
    i16x8.all_true

    # CHECK: i16x8.shl # encoding: [0xfd,0x65]
    i16x8.shl

    # CHECK: i16x8.shr_s # encoding: [0xfd,0x66]
    i16x8.shr_s

    # CHECK: i16x8.shr_u # encoding: [0xfd,0x67]
    i16x8.shr_u

    # CHECK: i16x8.add # encoding: [0xfd,0x68]
    i16x8.add

    # CHECK: i16x8.add_saturate_s # encoding: [0xfd,0x69]
    i16x8.add_saturate_s

    # CHECK: i16x8.add_saturate_u # encoding: [0xfd,0x6a]
    i16x8.add_saturate_u

    # CHECK: i16x8.sub # encoding: [0xfd,0x6b]
    i16x8.sub

    # CHECK: i16x8.sub_saturate_s # encoding: [0xfd,0x6c]
    i16x8.sub_saturate_s

    # CHECK: i16x8.sub_saturate_u # encoding: [0xfd,0x6d]
    i16x8.sub_saturate_u

    # CHECK: i16x8.mul # encoding: [0xfd,0x6e]
    i16x8.mul

    # CHECK: i16x8.min_s # encoding: [0xfd,0x6f]
    i16x8.min_s

    # CHECK: i16x8.min_u # encoding: [0xfd,0x70]
    i16x8.min_u

    # CHECK: i16x8.max_s # encoding: [0xfd,0x71]
    i16x8.max_s

    # CHECK: i16x8.max_u # encoding: [0xfd,0x72]
    i16x8.max_u

    # CHECK: i32x4.neg # encoding: [0xfd,0x73]
    i32x4.neg

    # CHECK: i32x4.any_true # encoding: [0xfd,0x74]
    i32x4.any_true

    # CHECK: i32x4.all_true # encoding: [0xfd,0x75]
    i32x4.all_true

    # CHECK: i32x4.shl # encoding: [0xfd,0x76]
    i32x4.shl

    # CHECK: i32x4.shr_s # encoding: [0xfd,0x77]
    i32x4.shr_s

    # CHECK: i32x4.shr_u # encoding: [0xfd,0x78]
    i32x4.shr_u

    # CHECK: i32x4.add # encoding: [0xfd,0x79]
    i32x4.add

    # CHECK: i32x4.sub # encoding: [0xfd,0x7c]
    i32x4.sub

    # CHECK: i32x4.mul # encoding: [0xfd,0x7f]
    i32x4.mul

    # CHECK: i32x4.min_s # encoding: [0xfd,0x80,0x01]
    i32x4.min_s

    # CHECK: i32x4.min_u # encoding: [0xfd,0x81,0x01]
    i32x4.min_u

    # CHECK: i32x4.max_s # encoding: [0xfd,0x82,0x01]
    i32x4.max_s

    # CHECK: i32x4.max_u # encoding: [0xfd,0x83,0x01]
    i32x4.max_u

    # CHECK: i64x2.neg # encoding: [0xfd,0x84,0x01]
    i64x2.neg

    # CHECK: i64x2.any_true # encoding: [0xfd,0x85,0x01]
    i64x2.any_true

    # CHECK: i64x2.all_true # encoding: [0xfd,0x86,0x01]
    i64x2.all_true

    # CHECK: i64x2.shl # encoding: [0xfd,0x87,0x01]
    i64x2.shl

    # CHECK: i64x2.shr_s # encoding: [0xfd,0x88,0x01]
    i64x2.shr_s

    # CHECK: i64x2.shr_u # encoding: [0xfd,0x89,0x01]
    i64x2.shr_u

    # CHECK: i64x2.add # encoding: [0xfd,0x8a,0x01]
    i64x2.add

    # CHECK: i64x2.sub # encoding: [0xfd,0x8d,0x01]
    i64x2.sub

    # CHECK: f32x4.abs # encoding: [0xfd,0x95,0x01]
    f32x4.abs

    # CHECK: f32x4.neg # encoding: [0xfd,0x96,0x01]
    f32x4.neg

    # CHECK: f32x4.sqrt # encoding: [0xfd,0x97,0x01]
    f32x4.sqrt

    # CHECK: f32x4.qfma # encoding: [0xfd,0x98,0x01]
    f32x4.qfma

    # CHECK: f32x4.qfms # encoding: [0xfd,0x99,0x01]
    f32x4.qfms

    # CHECK: f32x4.add # encoding: [0xfd,0x9a,0x01]
    f32x4.add

    # CHECK: f32x4.sub # encoding: [0xfd,0x9b,0x01]
    f32x4.sub

    # CHECK: f32x4.mul # encoding: [0xfd,0x9c,0x01]
    f32x4.mul

    # CHECK: f32x4.div # encoding: [0xfd,0x9d,0x01]
    f32x4.div

    # CHECK: f32x4.min # encoding: [0xfd,0x9e,0x01]
    f32x4.min

    # CHECK: f32x4.max # encoding: [0xfd,0x9f,0x01]
    f32x4.max

    # CHECK: f64x2.abs # encoding: [0xfd,0xa0,0x01]
    f64x2.abs

    # CHECK: f64x2.neg # encoding: [0xfd,0xa1,0x01]
    f64x2.neg

    # CHECK: f64x2.sqrt # encoding: [0xfd,0xa2,0x01]
    f64x2.sqrt

    # CHECK: f64x2.qfma # encoding: [0xfd,0xa3,0x01]
    f64x2.qfma

    # CHECK: f64x2.qfms # encoding: [0xfd,0xa4,0x01]
    f64x2.qfms

    # CHECK: f64x2.add # encoding: [0xfd,0xa5,0x01]
    f64x2.add

    # CHECK: f64x2.sub # encoding: [0xfd,0xa6,0x01]
    f64x2.sub

    # CHECK: f64x2.mul # encoding: [0xfd,0xa7,0x01]
    f64x2.mul

    # CHECK: f64x2.div # encoding: [0xfd,0xa8,0x01]
    f64x2.div

    # CHECK: f64x2.min # encoding: [0xfd,0xa9,0x01]
    f64x2.min

    # CHECK: f64x2.max # encoding: [0xfd,0xaa,0x01]
    f64x2.max

    # CHECK: i32x4.trunc_sat_f32x4_s # encoding: [0xfd,0xab,0x01]
    i32x4.trunc_sat_f32x4_s

    # CHECK: i32x4.trunc_sat_f32x4_u # encoding: [0xfd,0xac,0x01]
    i32x4.trunc_sat_f32x4_u

    # CHECK: i64x2.trunc_sat_f64x2_s # encoding: [0xfd,0xad,0x01]
    i64x2.trunc_sat_f64x2_s

    # CHECK: i64x2.trunc_sat_f64x2_u # encoding: [0xfd,0xae,0x01]
    i64x2.trunc_sat_f64x2_u

    # CHECK: f32x4.convert_i32x4_s # encoding: [0xfd,0xaf,0x01]
    f32x4.convert_i32x4_s

    # CHECK: f32x4.convert_i32x4_u # encoding: [0xfd,0xb0,0x01]
    f32x4.convert_i32x4_u

    # CHECK: f64x2.convert_i64x2_s # encoding: [0xfd,0xb1,0x01]
    f64x2.convert_i64x2_s

    # CHECK: f64x2.convert_i64x2_u # encoding: [0xfd,0xb2,0x01]
    f64x2.convert_i64x2_u

    # CHECK: v8x16.swizzle # encoding: [0xfd,0xc0,0x01]
    v8x16.swizzle

    # CHECK: v8x16.load_splat 48 # encoding: [0xfd,0xc2,0x01,0x00,0x30]
    v8x16.load_splat 48

    # CHECK: v16x8.load_splat 48 # encoding: [0xfd,0xc3,0x01,0x01,0x30]
    v16x8.load_splat 48

    # CHECK: v32x4.load_splat 48 # encoding: [0xfd,0xc4,0x01,0x02,0x30]
    v32x4.load_splat 48

    # CHECK: v64x2.load_splat 48 # encoding: [0xfd,0xc5,0x01,0x03,0x30]
    v64x2.load_splat 48

    # CHECK: i8x16.narrow_i16x8_s # encoding: [0xfd,0xc6,0x01]
    i8x16.narrow_i16x8_s

    # CHECK: i8x16.narrow_i16x8_u # encoding: [0xfd,0xc7,0x01]
    i8x16.narrow_i16x8_u

    # CHECK: i16x8.narrow_i32x4_s # encoding: [0xfd,0xc8,0x01]
    i16x8.narrow_i32x4_s

    # CHECK: i16x8.narrow_i32x4_u # encoding: [0xfd,0xc9,0x01]
    i16x8.narrow_i32x4_u

    # CHECK: i16x8.widen_low_i8x16_s # encoding: [0xfd,0xca,0x01]
    i16x8.widen_low_i8x16_s

    # CHECK: i16x8.widen_high_i8x16_s # encoding: [0xfd,0xcb,0x01]
    i16x8.widen_high_i8x16_s

    # CHECK: i16x8.widen_low_i8x16_u # encoding: [0xfd,0xcc,0x01]
    i16x8.widen_low_i8x16_u

    # CHECK: i16x8.widen_high_i8x16_u # encoding: [0xfd,0xcd,0x01]
    i16x8.widen_high_i8x16_u

    # CHECK: i32x4.widen_low_i16x8_s # encoding: [0xfd,0xce,0x01]
    i32x4.widen_low_i16x8_s

    # CHECK: i32x4.widen_high_i16x8_s # encoding: [0xfd,0xcf,0x01]
    i32x4.widen_high_i16x8_s

    # CHECK: i32x4.widen_low_i16x8_u # encoding: [0xfd,0xd0,0x01]
    i32x4.widen_low_i16x8_u

    # CHECK: i32x4.widen_high_i16x8_u # encoding: [0xfd,0xd1,0x01]
    i32x4.widen_high_i16x8_u

    # CHECK: i16x8.load8x8_s 32 # encoding: [0xfd,0xd2,0x01,0x03,0x20]
    i16x8.load8x8_s 32

    # CHECK: i16x8.load8x8_u 32 # encoding: [0xfd,0xd3,0x01,0x03,0x20]
    i16x8.load8x8_u 32

    # CHECK: i32x4.load16x4_s 32 # encoding: [0xfd,0xd4,0x01,0x03,0x20]
    i32x4.load16x4_s 32

    # CHECK: i32x4.load16x4_u 32 # encoding: [0xfd,0xd5,0x01,0x03,0x20]
    i32x4.load16x4_u 32

    # CHECK: i64x2.load32x2_s 32 # encoding: [0xfd,0xd6,0x01,0x03,0x20]
    i64x2.load32x2_s 32

    # CHECK: i64x2.load32x2_u 32 # encoding: [0xfd,0xd7,0x01,0x03,0x20]
    i64x2.load32x2_u 32

    # CHECK: v128.andnot # encoding: [0xfd,0xd8,0x01]
    v128.andnot

    # CHECK: i8x16.avgr_u # encoding: [0xfd,0xd9,0x01]
    i8x16.avgr_u

    # CHECK: i16x8.avgr_u # encoding: [0xfd,0xda,0x01]
    i16x8.avgr_u

    # CHECK: i32x4.dot_i16x8_s # encoding: [0xfd,0xdb,0x01]
    i32x4.dot_i16x8_s

    # CHECK: i8x16.bitmask # encoding: [0xfd,0xe4,0x01]
    i8x16.bitmask

    # CHECK: i16x8.bitmask # encoding: [0xfd,0xe5,0x01]
    i16x8.bitmask

    # CHECK: i32x4.bitmask # encoding: [0xfd,0xe6,0x01]
    i32x4.bitmask

    end_function
