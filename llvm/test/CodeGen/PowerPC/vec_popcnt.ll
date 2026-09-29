; Check the vecpopcnt* instructions that were added in P8
; In addition, check the conversions to/from the v2i64 VMX register that was also added in P8.
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s

declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>) nounwind readnone
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16>) nounwind readnone
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>) nounwind readnone
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>) nounwind readnone

define <16 x i8> @test_v16i8_v2i64(<2 x i64> %x) nounwind readnone {
       %tmp  = bitcast <2 x i64> %x to <16 x i8>;
       %vcnt = tail call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %tmp)
       ret <16 x i8> %vcnt
; CHECK: @test_v16i8_v2i64
; CHECK: vpopcntb 2, 2
; CHECK: blr
}

define <8 x i16> @test_v8i16_v2i64(<2 x i64> %x) nounwind readnone {
       %tmp = bitcast <2 x i64> %x to <8 x i16>
       %vcnt = tail call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %tmp)
       ret <8 x i16> %vcnt
; CHECK: @test_v8i16_v2i64
; CHECK: vpopcnth 2, 2
; CHECK: blr
}

define <4 x i32> @test_v4i32_v2i64(<2 x i64> %x) nounwind readnone {
       %tmp = bitcast <2 x i64> %x to <4 x i32>
       %vcnt = tail call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %tmp)
       ret <4 x i32> %vcnt
; CHECK: @test_v4i32_v2i64
; CHECK: vpopcntw 2, 2
; CHECK: blr
}

define <2 x i64> @test_v2i64_v2i64(<2 x i64> %x) nounwind readnone {
       %vcnt = tail call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %x)
       ret <2 x i64> %vcnt
; CHECK: @test_v2i64_v2i64
; CHECK: vpopcntd 2, 2
; CHECK: blr
}

define <2 x i64> @test_v2i64_v4i32(<4 x i32> %x) nounwind readnone {
       %tmp = bitcast <4 x i32> %x to <2 x i64>
       %vcnt = tail call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %tmp)
       ret <2 x i64> %vcnt
; CHECK: @test_v2i64_v4i32
; CHECK: vpopcntd 2, 2
; CHECK: blr
}


define <2 x i64> @test_v2i64_v8i16(<8 x i16> %x) nounwind readnone {
       %tmp = bitcast <8 x i16> %x to <2 x i64>
       %vcnt = tail call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %tmp)
       ret <2 x i64> %vcnt
; CHECK: @test_v2i64_v8i16
; CHECK: vpopcntd 2, 2
; CHECK: blr
}

define <2 x i64> @test_v2i64_v16i8(<16 x i8> %x) nounwind readnone {
       %tmp = bitcast <16 x i8> %x to <2 x i64>
       %vcnt = tail call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %tmp)
       ret <2 x i64> %vcnt
; CHECK: @test_v2i64_v16i8
; CHECK: vpopcntd 2, 2
; CHECK: blr
}
