# RUN: llvm-mc -show-encoding -triple=wasm32-unkown-unknown -mattr=+sign-ext,+simd128 < %s | FileCheck %s

    # CHECK: v128.const 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    # CHECK-SAME: # encoding: [0xfd,0x00,
    # CHECK-SAME: 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    # CHECK-SAME: 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f]
    v128.const 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

    # CHECK: v128.const 256, 770, 1284, 1798, 2312, 2826, 3340, 3854
    # CHECK-SAME: # encoding: [0xfd,0x00,
    # CHECK-SAME: 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    # CHECK-SAME: 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f]
    v128.const 256, 770, 1284, 1798, 2312, 2826, 3340, 3854

    # TODO(tlively): Fix assembler so v128.const works with 4xi32 and 2xi64

    # CHECK: v128.const 0x1.0402p-121, 0x1.0c0a08p-113,
    # CHECK-SAME:       0x1.14121p-105, 0x1.1c1a18p-97
    # CHECK-SAME: # encoding: [0xfd,0x00,
    # CHECK-SAME: 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    # CHECK-SAME: 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f]
    v128.const 0x1.0402p-121, 0x1.0c0a08p-113, 0x1.14121p-105, 0x1.1c1a18p-97

    # CHECK: v128.const 0x1.60504030201p-911, 0x1.e0d0c0b0a0908p-783
    # CHECK-SAME: # encoding: [0xfd,0x00,
    # CHECK-SAME: 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    # CHECK-SAME: 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f]
    v128.const 0x1.60504030201p-911, 0x1.e0d0c0b0a0908p-783

    # CHECK: v128.load 48:p2align=0 # encoding: [0xfd,0x01,0x00,0x30]
    v128.load 48

    # CHECK: v128.store 48:p2align=0 # encoding: [0xfd,0x02,0x00,0x30]
    v128.store 48

    # CHECK: i8x16.splat # encoding: [0xfd,0x03]
    i8x16.splat

    # CHECK: i16x8.splat # encoding: [0xfd,0x04]
    i16x8.splat

    # CHECK: i32x4.splat # encoding: [0xfd,0x05]
    i32x4.splat

    # CHECK: i64x2.splat # encoding: [0xfd,0x06]
    i64x2.splat

    # CHECK: f32x4.splat # encoding: [0xfd,0x07]
    f32x4.splat

    # CHECK: f64x2.splat # encoding: [0xfd,0x08]
    f64x2.splat

    # CHECK: i8x16.extract_lane_s 15 # encoding: [0xfd,0x09,0x0f]
    i8x16.extract_lane_s 15

    # CHECK: i8x16.extract_lane_u 15 # encoding: [0xfd,0x0a,0x0f]
    i8x16.extract_lane_u 15

    # CHECK: i16x8.extract_lane_s 7 # encoding: [0xfd,0x0b,0x07]
    i16x8.extract_lane_s 7

    # CHECK: i16x8.extract_lane_u 7 # encoding: [0xfd,0x0c,0x07]
    i16x8.extract_lane_u 7

    # CHECK: i32x4.extract_lane 3 # encoding: [0xfd,0x0d,0x03]
    i32x4.extract_lane 3

    # CHECK: i64x2.extract_lane 1 # encoding: [0xfd,0x0e,0x01]
    i64x2.extract_lane 1

    # CHECK: f32x4.extract_lane 3 # encoding: [0xfd,0x0f,0x03]
    f32x4.extract_lane 3

    # CHECK: f64x2.extract_lane 1 # encoding: [0xfd,0x10,0x01]
    f64x2.extract_lane 1

    # CHECK: i8x16.replace_lane 15 # encoding: [0xfd,0x11,0x0f]
    i8x16.replace_lane 15

    # CHECK: i16x8.replace_lane 7 # encoding: [0xfd,0x12,0x07]
    i16x8.replace_lane 7

    # CHECK: i32x4.replace_lane 3 # encoding: [0xfd,0x13,0x03]
    i32x4.replace_lane 3

    # CHECK: i64x2.replace_lane 1 # encoding: [0xfd,0x14,0x01]
    i64x2.replace_lane 1

    # CHECK: f32x4.replace_lane 3 # encoding: [0xfd,0x15,0x03]
    f32x4.replace_lane 3

    # CHECK: f64x2.replace_lane 1 # encoding: [0xfd,0x16,0x01]
    f64x2.replace_lane 1

    # CHECK: v8x16.shuffle 0, 17, 2, 19, 4, 21, 6, 23,
    # CHECK-SAME:          8, 25, 10, 27, 12, 29, 14, 31
    # CHECK-SAME: # encoding: [0xfd,0x17,
    # CHECK-SAME: 0x00,0x11,0x02,0x13,0x04,0x15,0x06,0x17,
    # CHECK-SAME: 0x08,0x19,0x0a,0x1b,0x0c,0x1d,0x0e,0x1f]
    v8x16.shuffle 0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31

    # CHECK: i8x16.add # encoding: [0xfd,0x18]
    i8x16.add

    # CHECK: i16x8.add # encoding: [0xfd,0x19]
    i16x8.add

    # CHECK: i32x4.add # encoding: [0xfd,0x1a]
    i32x4.add

    # CHECK: i64x2.add # encoding: [0xfd,0x1b]
    i64x2.add

    # CHECK: i8x16.sub # encoding: [0xfd,0x1c]
    i8x16.sub

    # CHECK: i16x8.sub # encoding: [0xfd,0x1d]
    i16x8.sub

    # CHECK: i32x4.sub # encoding: [0xfd,0x1e]
    i32x4.sub

    # CHECK: i64x2.sub # encoding: [0xfd,0x1f]
    i64x2.sub

    # CHECK: i8x16.mul # encoding: [0xfd,0x20]
    i8x16.mul

    # CHECK: i16x8.mul # encoding: [0xfd,0x21]
    i16x8.mul

    # CHECK: i32x4.mul # encoding: [0xfd,0x22]
    i32x4.mul

    # CHECK: i8x16.neg # encoding: [0xfd,0x24]
    i8x16.neg

    # CHECK: i16x8.neg # encoding: [0xfd,0x25]
    i16x8.neg

    # CHECK: i32x4.neg # encoding: [0xfd,0x26]
    i32x4.neg

    # CHECK: i64x2.neg # encoding: [0xfd,0x27]
    i64x2.neg

    # CHECK: i8x16.shl # encoding: [0xfd,0x30]
    i8x16.shl

    # CHECK: i16x8.shl # encoding: [0xfd,0x31]
    i16x8.shl

    # CHECK: i32x4.shl # encoding: [0xfd,0x32]
    i32x4.shl

    # CHECK: i64x2.shl # encoding: [0xfd,0x33]
    i64x2.shl

    # CHECK: i8x16.shr_s # encoding: [0xfd,0x34]
    i8x16.shr_s

    # CHECK: i8x16.shr_u # encoding: [0xfd,0x35]
    i8x16.shr_u

    # CHECK: i16x8.shr_s # encoding: [0xfd,0x36]
    i16x8.shr_s

    # CHECK: i16x8.shr_u # encoding: [0xfd,0x37]
    i16x8.shr_u

    # CHECK: i32x4.shr_s # encoding: [0xfd,0x38]
    i32x4.shr_s

    # CHECK: i32x4.shr_u # encoding: [0xfd,0x39]
    i32x4.shr_u

    # CHECK: i64x2.shr_s # encoding: [0xfd,0x3a]
    i64x2.shr_s

    # CHECK: i64x2.shr_u # encoding: [0xfd,0x3b]
    i64x2.shr_u

    # CHECK: v128.and # encoding: [0xfd,0x3c]
    v128.and

    # CHECK: v128.or # encoding: [0xfd,0x3d]
    v128.or

    # CHECK: v128.xor # encoding: [0xfd,0x3e]
    v128.xor

    # CHECK: v128.not # encoding: [0xfd,0x3f]
    v128.not

    # CHECK: i8x16.eq # encoding: [0xfd,0x49]
    i8x16.eq

    # CHECK: i16x8.eq # encoding: [0xfd,0x4a]
    i16x8.eq

    # CHECK: i32x4.eq # encoding: [0xfd,0x4b]
    i32x4.eq

    # CHECK: f32x4.eq # encoding: [0xfd,0x4d]
    f32x4.eq

    # CHECK: f64x2.eq # encoding: [0xfd,0x4e]
    f64x2.eq

    # CHECK: i8x16.ne # encoding: [0xfd,0x4f]
    i8x16.ne

    # CHECK: i16x8.ne # encoding: [0xfd,0x50]
    i16x8.ne

    # CHECK: i32x4.ne # encoding: [0xfd,0x51]
    i32x4.ne

    # CHECK: f32x4.ne # encoding: [0xfd,0x53]
    f32x4.ne

    # CHECK: f64x2.ne # encoding: [0xfd,0x54]
    f64x2.ne

    # CHECK: i8x16.lt_s # encoding: [0xfd,0x55]
    i8x16.lt_s

    # CHECK: i8x16.lt_u # encoding: [0xfd,0x56]
    i8x16.lt_u

    # CHECK: i16x8.lt_s # encoding: [0xfd,0x57]
    i16x8.lt_s

    # CHECK: i16x8.lt_u # encoding: [0xfd,0x58]
    i16x8.lt_u

    # CHECK: i32x4.lt_s # encoding: [0xfd,0x59]
    i32x4.lt_s

    # CHECK: i32x4.lt_u # encoding: [0xfd,0x5a]
    i32x4.lt_u

    # CHECK: f32x4.lt # encoding: [0xfd,0x5d]
    f32x4.lt

    # CHECK: f64x2.lt # encoding: [0xfd,0x5e]
    f64x2.lt

    # CHECK: i8x16.le_s # encoding: [0xfd,0x5f]
    i8x16.le_s

    # CHECK: i8x16.le_u # encoding: [0xfd,0x60]
    i8x16.le_u

    # CHECK: i16x8.le_s # encoding: [0xfd,0x61]
    i16x8.le_s

    # CHECK: i16x8.le_u # encoding: [0xfd,0x62]
    i16x8.le_u

    # CHECK: i32x4.le_s # encoding: [0xfd,0x63]
    i32x4.le_s

    # CHECK: i32x4.le_u # encoding: [0xfd,0x64]
    i32x4.le_u

    # CHECK: f32x4.le # encoding: [0xfd,0x67]
    f32x4.le

    # CHECK: f64x2.le # encoding: [0xfd,0x68]
    f64x2.le

    # CHECK: i8x16.gt_s # encoding: [0xfd,0x69]
    i8x16.gt_s

    # CHECK: i8x16.gt_u # encoding: [0xfd,0x6a]
    i8x16.gt_u

    # CHECK: i16x8.gt_s # encoding: [0xfd,0x6b]
    i16x8.gt_s

    # CHECK: i16x8.gt_u # encoding: [0xfd,0x6c]
    i16x8.gt_u

    # CHECK: i32x4.gt_s # encoding: [0xfd,0x6d]
    i32x4.gt_s

    # CHECK: i32x4.gt_u # encoding: [0xfd,0x6e]
    i32x4.gt_u

    # CHECK: f32x4.gt # encoding: [0xfd,0x71]
    f32x4.gt

    # CHECK: f64x2.gt # encoding: [0xfd,0x72]
    f64x2.gt

    # CHECK: i8x16.ge_s # encoding: [0xfd,0x73]
    i8x16.ge_s

    # CHECK: i8x16.ge_u # encoding: [0xfd,0x74]
    i8x16.ge_u

    # CHECK: i16x8.ge_s # encoding: [0xfd,0x75]
    i16x8.ge_s

    # CHECK: i16x8.ge_u # encoding: [0xfd,0x76]
    i16x8.ge_u

    # CHECK: i32x4.ge_s # encoding: [0xfd,0x77]
    i32x4.ge_s

    # CHECK: i32x4.ge_u # encoding: [0xfd,0x78]
    i32x4.ge_u

    # CHECK: f32x4.ge # encoding: [0xfd,0x7b]
    f32x4.ge

    # CHECK: f64x2.ge # encoding: [0xfd,0x7c]
    f64x2.ge

    # CHECK: f32x4.neg # encoding: [0xfd,0x7d]
    f32x4.neg

    # CHECK: f64x2.neg # encoding: [0xfd,0x7e]
    f64x2.neg

    # CHECK: f32x4.abs # encoding: [0xfd,0x7f]
    f32x4.abs

    # CHECK: f64x2.abs # encoding: [0xfd,0x80]
    f64x2.abs

    # CHECK: f32x4.add # encoding: [0xfd,0x85]
    f32x4.add

    # CHECK: f64x2.add # encoding: [0xfd,0x86]
    f64x2.add

    # CHECK: f32x4.sub # encoding: [0xfd,0x87]
    f32x4.sub

    # CHECK: f64x2.sub # encoding: [0xfd,0x88]
    f64x2.sub

    # CHECK: f32x4.div # encoding: [0xfd,0x89]
    f32x4.div

    # CHECK: f64x2.div # encoding: [0xfd,0x8a]
    f64x2.div

    # CHECK: f32x4.mul # encoding: [0xfd,0x8b]
    f32x4.mul

    # CHECK: f64x2.mul # encoding: [0xfd,0x8c]
    f64x2.mul

    # CHECK: f32x4.sqrt # encoding: [0xfd,0x8d]
    f32x4.sqrt

    # CHECK: f64x2.sqrt # encoding: [0xfd,0x8e]
    f64x2.sqrt

    end_function
