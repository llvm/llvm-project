; RUN: llc -mtriple=xtensa -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=XTENSA %s

declare i8 @llvm.cttz.i8(i8, i1)
declare i16 @llvm.cttz.i16(i16, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i8 @llvm.ctlz.i8(i8, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i64 @llvm.ctlz.i64(i64, i1)
declare i8 @llvm.ctpop.i8(i8)
declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)
declare i64 @llvm.ctpop.i64(i64)

define i8 @test_cttz_i8(i8 %a) nounwind {
; XTENSA-LABEL: test_cttz_i8:
; XTENSA:         movi a9, 255
; XTENSA-NEXT:    and a10, a2, a9
; XTENSA-NEXT:    movi a8, 8
; XTENSA-NEXT:    beqz a10, .LBB0_2
; XTENSA-NEXT:    j .LBB0_1
; XTENSA-NEXT:  .LBB0_1: # %cond.false
; XTENSA-NEXT:    movi a8, -1
; XTENSA-NEXT:    xor a8, a2, a8
; XTENSA-NEXT:    addi a10, a2, -1
; XTENSA-NEXT:    and a8, a8, a10
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI0_0
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI0_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI0_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a8, a8, 0, 6
; XTENSA-NEXT:  .LBB0_2: # %cond.end
; XTENSA-NEXT:    or a2, a8, a8
; XTENSA-NEXT:    ret
  %tmp = call i8 @llvm.cttz.i8(i8 %a, i1 false)
  ret i8 %tmp
}

define i16 @test_cttz_i16(i16 %a) nounwind {
; XTENSA-LABEL: test_cttz_i16:
; XTENSA:         l32r a9, .LCPI1_0
; XTENSA-NEXT:    and a10, a2, a9
; XTENSA-NEXT:    movi a8, 16
; XTENSA-NEXT:    beqz a10, .LBB1_2
; XTENSA-NEXT:    j .LBB1_1
; XTENSA-NEXT:  .LBB1_1: # %cond.false
; XTENSA-NEXT:    movi a8, -1
; XTENSA-NEXT:    xor a8, a2, a8
; XTENSA-NEXT:    addi a10, a2, -1
; XTENSA-NEXT:    and a8, a8, a10
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI1_1
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI1_2
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI1_3
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a8, a8, 0, 6
; XTENSA-NEXT:  .LBB1_2: # %cond.end
; XTENSA-NEXT:    or a2, a8, a8
; XTENSA-NEXT:    ret
  %tmp = call i16 @llvm.cttz.i16(i16 %a, i1 false)
  ret i16 %tmp
}

define i32 @test_cttz_i32(i32 %a) nounwind {
; XTENSA-LABEL: test_cttz_i32:
; XTENSA:         movi a8, 32
; XTENSA-NEXT:    beqz a2, .LBB2_2
; XTENSA-NEXT:    j .LBB2_1
; XTENSA-NEXT:  .LBB2_1: # %cond.false
; XTENSA-NEXT:    movi a8, -1
; XTENSA-NEXT:    xor a8, a2, a8
; XTENSA-NEXT:    addi a9, a2, -1
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI2_0
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI2_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI2_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a8, a8, 0, 6
; XTENSA-NEXT:  .LBB2_2: # %cond.end
; XTENSA-NEXT:    or a2, a8, a8
; XTENSA-NEXT:    ret
  %tmp = call i32 @llvm.cttz.i32(i32 %a, i1 false)
  ret i32 %tmp
}

define i8 @test_cttz_i8_zero_undef(i8 %a) nounwind {
; XTENSA-LABEL: test_cttz_i8_zero_undef:
; XTENSA:         movi a8, -1
; XTENSA-NEXT:    xor a8, a2, a8
; XTENSA-NEXT:    addi a9, a2, -1
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    movi a9, 255
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI3_0
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI3_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI3_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:    ret
  %tmp = call i8 @llvm.cttz.i8(i8 %a, i1 true)
  ret i8 %tmp
}

define i16 @test_cttz_i16_zero_undef(i16 %a) nounwind {
; XTENSA-LABEL: test_cttz_i16_zero_undef:
; XTENSA:         movi a8, -1
; XTENSA-NEXT:    xor a8, a2, a8
; XTENSA-NEXT:    addi a9, a2, -1
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI4_0
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI4_1
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI4_2
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI4_3
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:    ret
  %tmp = call i16 @llvm.cttz.i16(i16 %a, i1 true)
  ret i16 %tmp
}

define i32 @test_cttz_i32_zero_undef(i32 %a) nounwind {
; XTENSA-LABEL: test_cttz_i32_zero_undef:
; XTENSA:         movi a8, -1
; XTENSA-NEXT:    xor a8, a2, a8
; XTENSA-NEXT:    addi a9, a2, -1
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI5_0
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI5_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI5_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:    ret
  %tmp = call i32 @llvm.cttz.i32(i32 %a, i1 true)
  ret i32 %tmp
}

define i8 @test_ctlz_i8(i8 %a) nounwind {
; XTENSA-LABEL: test_ctlz_i8:
; XTENSA:         movi a9, 255
; XTENSA-NEXT:    and a10, a2, a9
; XTENSA-NEXT:    movi a8, 8
; XTENSA-NEXT:    beqz a10, .LBB6_2
; XTENSA-NEXT:    j .LBB6_1
; XTENSA-NEXT:  .LBB6_1: # %cond.false
; XTENSA-NEXT:    movi a8, 254
; XTENSA-NEXT:    and a8, a2, a8
; XTENSA-NEXT:    srli a8, a8, 1
; XTENSA-NEXT:    or a8, a2, a8
; XTENSA-NEXT:    movi a10, 252
; XTENSA-NEXT:    and a10, a8, a10
; XTENSA-NEXT:    srli a10, a10, 2
; XTENSA-NEXT:    or a8, a8, a10
; XTENSA-NEXT:    movi a10, 240
; XTENSA-NEXT:    and a10, a8, a10
; XTENSA-NEXT:    srli a10, a10, 4
; XTENSA-NEXT:    or a8, a8, a10
; XTENSA-NEXT:    movi a10, -1
; XTENSA-NEXT:    xor a8, a8, a10
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI6_0
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI6_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI6_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a8, a8, 0, 6
; XTENSA-NEXT:  .LBB6_2: # %cond.end
; XTENSA-NEXT:    or a2, a8, a8
; XTENSA-NEXT:    ret
  %tmp = call i8 @llvm.ctlz.i8(i8 %a, i1 false)
  ret i8 %tmp
}

define i16 @test_ctlz_i16(i16 %a) nounwind {
; XTENSA-LABEL: test_ctlz_i16:
; XTENSA:         l32r a9, .LCPI7_0
; XTENSA-NEXT:    and a10, a2, a9
; XTENSA-NEXT:    movi a8, 16
; XTENSA-NEXT:    beqz a10, .LBB7_2
; XTENSA-NEXT:    j .LBB7_1
; XTENSA-NEXT:  .LBB7_1: # %cond.false
; XTENSA-NEXT:    l32r a8, .LCPI7_1
; XTENSA-NEXT:    and a8, a2, a8
; XTENSA-NEXT:    srli a8, a8, 1
; XTENSA-NEXT:    or a8, a2, a8
; XTENSA-NEXT:    l32r a10, .LCPI7_2
; XTENSA-NEXT:    and a10, a8, a10
; XTENSA-NEXT:    srli a10, a10, 2
; XTENSA-NEXT:    or a8, a8, a10
; XTENSA-NEXT:    l32r a10, .LCPI7_3
; XTENSA-NEXT:    and a10, a8, a10
; XTENSA-NEXT:    srli a10, a10, 4
; XTENSA-NEXT:    or a8, a8, a10
; XTENSA-NEXT:    l32r a10, .LCPI7_4
; XTENSA-NEXT:    and a10, a8, a10
; XTENSA-NEXT:    srli a10, a10, 8
; XTENSA-NEXT:    or a8, a8, a10
; XTENSA-NEXT:    movi a10, -1
; XTENSA-NEXT:    xor a8, a8, a10
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI7_5
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI7_6
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI7_7
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a8, a8, 0, 6
; XTENSA-NEXT:  .LBB7_2: # %cond.end
; XTENSA-NEXT:    or a2, a8, a8
; XTENSA-NEXT:    ret
  %tmp = call i16 @llvm.ctlz.i16(i16 %a, i1 false)
  ret i16 %tmp
}

define i32 @test_ctlz_i32(i32 %a) nounwind {
; XTENSA-LABEL: test_ctlz_i32:
; XTENSA:         or a8, a2, a2
; XTENSA-NEXT:    movi a2, 32
; XTENSA-NEXT:    beqz a8, .LBB8_2
; XTENSA-NEXT:    j .LBB8_1
; XTENSA-NEXT:  .LBB8_1: # %cond.false
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 2
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    movi a9, 16
; XTENSA-NEXT:    ssr a9
; XTENSA-NEXT:    srl a9, a8
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    movi a9, -1
; XTENSA-NEXT:    xor a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI8_0
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI8_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI8_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:  .LBB8_2: # %cond.end
; XTENSA-NEXT:    ret
  %tmp = call i32 @llvm.ctlz.i32(i32 %a, i1 false)
  ret i32 %tmp
}

define i8 @test_ctlz_i8_zero_undef(i8 %a) nounwind {
; XTENSA-LABEL: test_ctlz_i8_zero_undef:
; XTENSA:         movi a8, 254
; XTENSA-NEXT:    and a8, a2, a8
; XTENSA-NEXT:    srli a8, a8, 1
; XTENSA-NEXT:    or a8, a2, a8
; XTENSA-NEXT:    movi a9, 252
; XTENSA-NEXT:    and a9, a8, a9
; XTENSA-NEXT:    srli a9, a9, 2
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    movi a9, 240
; XTENSA-NEXT:    and a9, a8, a9
; XTENSA-NEXT:    srli a9, a9, 4
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    movi a9, -1
; XTENSA-NEXT:    xor a8, a8, a9
; XTENSA-NEXT:    movi a9, 255
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI9_0
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI9_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI9_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:    ret
  %tmp = call i8 @llvm.ctlz.i8(i8 %a, i1 true)
  ret i8 %tmp
}

define i16 @test_ctlz_i16_zero_undef(i16 %a) nounwind {
; XTENSA-LABEL: test_ctlz_i16_zero_undef:
; XTENSA:         l32r a8, .LCPI10_0
; XTENSA-NEXT:    and a8, a2, a8
; XTENSA-NEXT:    srli a8, a8, 1
; XTENSA-NEXT:    or a8, a2, a8
; XTENSA-NEXT:    l32r a9, .LCPI10_1
; XTENSA-NEXT:    and a9, a8, a9
; XTENSA-NEXT:    srli a9, a9, 2
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI10_2
; XTENSA-NEXT:    and a9, a8, a9
; XTENSA-NEXT:    srli a9, a9, 4
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI10_3
; XTENSA-NEXT:    and a9, a8, a9
; XTENSA-NEXT:    srli a9, a9, 8
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    movi a9, -1
; XTENSA-NEXT:    xor a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI10_4
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI10_5
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI10_6
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI10_7
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:    ret
  %tmp = call i16 @llvm.ctlz.i16(i16 %a, i1 true)
  ret i16 %tmp
}

define i32 @test_ctlz_i32_zero_undef(i32 %a) nounwind {
; XTENSA-LABEL: test_ctlz_i32_zero_undef:
; XTENSA:         srli a8, a2, 1
; XTENSA-NEXT:    or a8, a2, a8
; XTENSA-NEXT:    srli a9, a8, 2
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    movi a9, 16
; XTENSA-NEXT:    ssr a9
; XTENSA-NEXT:    srl a9, a8
; XTENSA-NEXT:    or a8, a8, a9
; XTENSA-NEXT:    movi a9, -1
; XTENSA-NEXT:    xor a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI11_0
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI11_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI11_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:    ret
  %tmp = call i32 @llvm.ctlz.i32(i32 %a, i1 true)
  ret i32 %tmp
}

define i8 @test_ctpop_i8(i8 %a) nounwind {
; XTENSA-LABEL: test_ctpop_i8:
; XTENSA:         movi a8, 255
; XTENSA-NEXT:    and a8, a2, a8
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI12_0
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI12_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI12_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:    ret
  %1 = call i8 @llvm.ctpop.i8(i8 %a)
  ret i8 %1
}

define i16 @test_ctpop_i16(i16 %a) nounwind {
; XTENSA-LABEL: test_ctpop_i16:
; XTENSA:         l32r a8, .LCPI13_0
; XTENSA-NEXT:    and a8, a2, a8
; XTENSA-NEXT:    srli a9, a8, 1
; XTENSA-NEXT:    l32r a10, .LCPI13_1
; XTENSA-NEXT:    and a9, a9, a10
; XTENSA-NEXT:    sub a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI13_2
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI13_3
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:    ret
  %1 = call i16 @llvm.ctpop.i16(i16 %a)
  ret i16 %1
}

define i32 @test_ctpop_i32(i32 %a) nounwind {
; XTENSA-LABEL: test_ctpop_i32:
; XTENSA:         srli a8, a2, 1
; XTENSA-NEXT:    l32r a9, .LCPI14_0
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    sub a8, a2, a8
; XTENSA-NEXT:    l32r a9, .LCPI14_1
; XTENSA-NEXT:    and a10, a8, a9
; XTENSA-NEXT:    srli a8, a8, 2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    add a8, a10, a8
; XTENSA-NEXT:    srli a9, a8, 4
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    l32r a9, .LCPI14_2
; XTENSA-NEXT:    and a8, a8, a9
; XTENSA-NEXT:    srli a9, a8, 8
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a9, a8, 16, 5
; XTENSA-NEXT:    add a8, a8, a9
; XTENSA-NEXT:    extui a2, a8, 0, 6
; XTENSA-NEXT:    ret
  %1 = call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %1
}
