; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx512f,+egpr -show-mc-encoding | FileCheck --check-prefix=EGPR %s

define <16 x i32> @kmovkk(ptr %base, <16 x i32> %ind, i16 %mask) {
; EGPR: kmovq   %k1, %k2                        # EVEX TO VEX Compression encoding: [0xc4,0xe1,0xf8,0x90,0xd1]
  %broadcast.splatinsert = insertelement <16 x ptr> undef, ptr %base, i32 0
  %broadcast.splat = shufflevector <16 x ptr> %broadcast.splatinsert, <16 x ptr> undef, <16 x i32> zeroinitializer
  %gep.random = getelementptr i32, <16 x ptr> %broadcast.splat, <16 x i32> %ind
  %imask = bitcast i16 %mask to <16 x i1>
  %gt1 = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %gep.random, i32 4, <16 x i1> %imask, <16 x i32>undef)
  %gt2 = call <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr> %gep.random, i32 4, <16 x i1> %imask, <16 x i32>%gt1)
  %res = add <16 x i32> %gt1, %gt2
  ret <16 x i32> %res
}
declare <16 x i32> @llvm.masked.gather.v16i32.v16p0(<16 x ptr>, i32, <16 x i1>, <16 x i32>)
