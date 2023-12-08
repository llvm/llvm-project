; check AVX2 instructions that are disabled in case avx512VL/avx512BW present

; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=corei7-avx                             -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=core-avx2 -mattr=+avx2                 -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=knl                                    -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=knl  -mattr=+avx512vl                  -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=knl  -mattr=+avx512bw                  -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=knl  -mattr=+avx512vl -mattr=+avx512bw -o /dev/null
; RUN: llc < %s -mtriple=x86_64-apple-darwin -show-mc-encoding -mcpu=skx                                    -o /dev/null

define <4 x i64> @vpand_256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = and <4 x i64> %a2, %b
  ret <4 x i64> %x
}

define <2 x i64> @vpand_128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %x = and <2 x i64> %a2, %b
  ret <2 x i64> %x
}

define <4 x i64> @vpandn_256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %y = xor <4 x i64> %a2, <i64 -1, i64 -1, i64 -1, i64 -1>
  %x = and <4 x i64> %a, %y
  ret <4 x i64> %x
}

define <2 x i64> @vpandn_128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %y = xor <2 x i64> %a2, <i64 -1, i64 -1>
  %x = and <2 x i64> %a, %y
  ret <2 x i64> %x
}

define <4 x i64> @vpor_256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = or <4 x i64> %a2, %b
  ret <4 x i64> %x
}

define <4 x i64> @vpxor_256(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = xor <4 x i64> %a2, %b
  ret <4 x i64> %x
}

define <2 x i64> @vpor_128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %x = or <2 x i64> %a2, %b
  ret <2 x i64> %x
}

define <2 x i64> @vpxor_128(<2 x i64> %a, <2 x i64> %b) nounwind uwtable readnone ssp {
  ; Force the execution domain with an add.
  %a2 = add <2 x i64> %a, <i64 1, i64 1>
  %x = xor <2 x i64> %a2, %b
  ret <2 x i64> %x
}

define <4 x i64> @test_vpaddq_256(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = add <4 x i64> %i, %j
  ret <4 x i64> %x
}

define <8 x i32> @test_vpaddd_256(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = add <8 x i32> %i, %j
  ret <8 x i32> %x
}

define <16 x i16> @test_vpaddw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = add <16 x i16> %i, %j
  ret <16 x i16> %x
}

define <32 x i8> @test_vpaddb_256(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = add <32 x i8> %i, %j
  ret <32 x i8> %x
}

define <4 x i64> @test_vpsubq_256(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = sub <4 x i64> %i, %j
  ret <4 x i64> %x
}

define <8 x i32> @test_vpsubd_256(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = sub <8 x i32> %i, %j
  ret <8 x i32> %x
}

define <16 x i16> @test_vpsubw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = sub <16 x i16> %i, %j
  ret <16 x i16> %x
}

define <32 x i8> @test_vpsubb_256(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = sub <32 x i8> %i, %j
  ret <32 x i8> %x
}

define <16 x i16> @test_vpmullw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = mul <16 x i16> %i, %j
  ret <16 x i16> %x
}

define <8 x i32> @test_vpcmpgtd_256(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %bincmp = icmp slt <8 x i32> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %x
}

define <32 x i8> @test_vpcmpeqb_256(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %bincmp = icmp eq <32 x i8> %i, %j
  %x = sext <32 x i1> %bincmp to <32 x i8>
  ret <32 x i8> %x
}

define <16 x i16> @test_vpcmpeqw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %bincmp = icmp eq <16 x i16> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i16>
  ret <16 x i16> %x
}

define <32 x i8> @test_vpcmpgtb_256(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %bincmp = icmp slt <32 x i8> %i, %j
  %x = sext <32 x i1> %bincmp to <32 x i8>
  ret <32 x i8> %x
}

define <16 x i16> @test_vpcmpgtw_256(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %bincmp = icmp slt <16 x i16> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i16>
  ret <16 x i16> %x
}

define <8 x i32> @test_vpcmpeqd_256(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %bincmp = icmp eq <8 x i32> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %x
}

define <2 x i64> @test_vpaddq_128(<2 x i64> %i, <2 x i64> %j) nounwind readnone {
  %x = add <2 x i64> %i, %j
  ret <2 x i64> %x
}

define <4 x i32> @test_vpaddd_128(<4 x i32> %i, <4 x i32> %j) nounwind readnone {
  %x = add <4 x i32> %i, %j
  ret <4 x i32> %x
}

define <8 x i16> @test_vpaddw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %x = add <8 x i16> %i, %j
  ret <8 x i16> %x
}

define <16 x i8> @test_vpaddb_128(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %x = add <16 x i8> %i, %j
  ret <16 x i8> %x
}

define <2 x i64> @test_vpsubq_128(<2 x i64> %i, <2 x i64> %j) nounwind readnone {
  %x = sub <2 x i64> %i, %j
  ret <2 x i64> %x
}

define <4 x i32> @test_vpsubd_128(<4 x i32> %i, <4 x i32> %j) nounwind readnone {
  %x = sub <4 x i32> %i, %j
  ret <4 x i32> %x
}

define <8 x i16> @test_vpsubw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %x = sub <8 x i16> %i, %j
  ret <8 x i16> %x
}

define <16 x i8> @test_vpsubb_128(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %x = sub <16 x i8> %i, %j
  ret <16 x i8> %x
}

define <8 x i16> @test_vpmullw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %x = mul <8 x i16> %i, %j
  ret <8 x i16> %x
}

define <8 x i16> @test_vpcmpgtw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %bincmp = icmp slt <8 x i16> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i16>
  ret <8 x i16> %x
}

define <16 x i8> @test_vpcmpgtb_128(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %bincmp = icmp slt <16 x i8> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i8>
  ret <16 x i8> %x
}

define <8 x i16> @test_vpcmpeqw_128(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %bincmp = icmp eq <8 x i16> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i16>
  ret <8 x i16> %x
}

define <16 x i8> @test_vpcmpeqb_128(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %bincmp = icmp eq <16 x i8> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i8>
  ret <16 x i8> %x
}

define <8 x i16> @shuffle_v8i16_vpalignr(<8 x i16> %a, <8 x i16> %b) {
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  ret <8 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_vpalignr(<16 x i16> %a, <16 x i16> %b) {
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 31, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <16 x i16> %shuffle
}

define <16 x i8> @shuffle_v16i8_vpalignr(<16 x i8> %a, <16 x i8> %b) {
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <16 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_vpalignr(<32 x i8> %a, <32 x i8> %b) {
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 undef, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 63, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <32 x i8> %shuffle
}

define <2 x i64> @shuffle_v2i64_vpalignr(<2 x i64> %a, <2 x i64> %b) {
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %shuffle
}

define <4 x i32> @shuffle_v4i32_vpalignr(<4 x i32> %a, <4 x i32> %b) {
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 7, i32 0, i32 1, i32 2>
  ret <4 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_vpalignr(<8 x i32> %a, <8 x i32> %b) {
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 11, i32 0, i32 1, i32 2, i32 15, i32 4, i32 5, i32 6>
  ret <8 x i32> %shuffle
}

define <4 x double> @shuffle_v4f64_5163(<4 x double> %a, <4 x double> %b) {
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 5, i32 1, i32 6, i32 3>
  ret <4 x double> %shuffle
}

define <2 x double> @shuffle_v2f64_bitcast_1z(<2 x double> %a) {
  %shuffle64 = shufflevector <2 x double> %a, <2 x double> zeroinitializer, <2 x i32> <i32 2, i32 1>
  %bitcast32 = bitcast <2 x double> %shuffle64 to <4 x float>
  %shuffle32 = shufflevector <4 x float> %bitcast32, <4 x float> undef, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
  %bitcast64 = bitcast <4 x float> %shuffle32 to <2 x double>
  ret <2 x double> %bitcast64
}

define <16 x i16> @shuffle_v16i16_zz_zz_zz_zz_zz_zz_zz_16_zz_zz_zz_zz_zz_zz_zz_24(<16 x i16> %a) {
  %shuffle = shufflevector <16 x i16> zeroinitializer, <16 x i16> %a, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 24>
  ret <16 x i16> %shuffle
}

define i64 @extract_v2i64(<2 x i64> %x, ptr %dst) {
  %r1 = extractelement <2 x i64> %x, i32 0
  %r2 = extractelement <2 x i64> %x, i32 1
  store i64 %r2, ptr %dst, align 1
  ret i64 %r1
}

define i32 @extract_v4i32(<4 x i32> %x, ptr %dst) {
  %r1 = extractelement <4 x i32> %x, i32 1
  %r2 = extractelement <4 x i32> %x, i32 3
  store i32 %r2, ptr %dst, align 1
  ret i32 %r1
}

define i16 @extract_v8i16(<8 x i16> %x, ptr %dst) {
  %r1 = extractelement <8 x i16> %x, i32 1
  %r2 = extractelement <8 x i16> %x, i32 3
  store i16 %r2, ptr %dst, align 1
  ret i16 %r1
}

define i8 @extract_v16i8(<16 x i8> %x, ptr %dst) {
  %r1 = extractelement <16 x i8> %x, i32 1
  %r2 = extractelement <16 x i8> %x, i32 3
  store i8 %r2, ptr %dst, align 1
  ret i8 %r1
}

define <2 x i64> @insert_v2i64(<2 x i64> %x, i64 %y , ptr %ptr) {
  %val = load i64, ptr %ptr
  %r1 = insertelement <2 x i64> %x, i64 %val, i32 1
  %r2 = insertelement <2 x i64> %r1, i64 %y, i32 3
  ret <2 x i64> %r2
}

define <4 x i32> @insert_v4i32(<4 x i32> %x, i32 %y, ptr %ptr) {
  %val = load i32, ptr %ptr
  %r1 = insertelement <4 x i32> %x, i32 %val, i32 1
  %r2 = insertelement <4 x i32> %r1, i32 %y, i32 3
  ret <4 x i32> %r2
}

define <8 x i16> @insert_v8i16(<8 x i16> %x, i16 %y, ptr %ptr) {
  %val = load i16, ptr %ptr
  %r1 = insertelement <8 x i16> %x, i16 %val, i32 1
  %r2 = insertelement <8 x i16> %r1, i16 %y, i32 5
  ret <8 x i16> %r2
}

define <16 x i8> @insert_v16i8(<16 x i8> %x, i8 %y, ptr %ptr) {
  %val = load i8, ptr %ptr
  %r1 = insertelement <16 x i8> %x, i8 %val, i32 3
  %r2 = insertelement <16 x i8> %r1, i8 %y, i32 10
  ret <16 x i8> %r2
}

define <4 x i32> @shuffle_v4i32_0451(<4 x i32> %a, <4 x i32> %b) {
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 5, i32 1>
  ret <4 x i32> %shuffle
}

define <4 x i32> @shuffle_v4i32_0142(<4 x i32> %a, <4 x i32> %b) {
 %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 2>
  ret <4 x i32> %shuffle
}

define <16 x i8> @shuffle_v16i8_0101010101010101(<16 x i8> %a, <16 x i8> %b) {
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  ret <16 x i8> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <8 x float> @shuffle_v8f32_11335577(<8 x float> %a, <8 x float> %b) {
; vmovshdup 256 test
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x float> %shuffle
}

define <4 x float> @shuffle_v4f32_1133(<4 x float> %a, <4 x float> %b) {
; vmovshdup 128 test
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  ret <4 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00224466(<8 x float> %a, <8 x float> %b) {
; vmovsldup 256 test
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x float> %shuffle
}

define <4 x float> @shuffle_v4f32_0022(<4 x float> %a, <4 x float> %b) {
; vmovsldup 128 test
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x float> %shuffle
}

define <2 x double> @insert_mem_lo_v2f64(ptr %ptr, <2 x double> %b) {
  %a = load double, ptr %ptr
  %v = insertelement <2 x double> undef, double %a, i32 0
  %shuffle = shufflevector <2 x double> %v, <2 x double> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %shuffle
}

define <2 x double> @insert_mem_hi_v2f64(ptr %ptr, <2 x double> %b) {
  %a = load double, ptr %ptr
  %v = insertelement <2 x double> undef, double %a, i32 0
  %shuffle = shufflevector <2 x double> %v, <2 x double> %b, <2 x i32> <i32 2, i32 0>
  ret <2 x double> %shuffle
}

define void @store_floats(<4 x float> %x, ptr %p) {
  %a = fadd <4 x float> %x, %x
  %b = shufflevector <4 x float> %a, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  %c = bitcast <2 x float> %b to i64
  store i64 %c, ptr %p
  ret void
}

define void @store_double(<2 x double> %x, ptr %p) {
  %a = fadd <2 x double> %x, %x
  %b = extractelement <2 x double> %a, i32 0
  %c = bitcast double %b to i64
  store i64 %c, ptr %p
  ret void
}

define void @store_h_double(<2 x double> %x, ptr %p) {
  %a = fadd <2 x double> %x, %x
  %b = extractelement <2 x double> %a, i32 1
  %c = bitcast double %b to i64
  store i64 %c, ptr %p
  ret void
}

define <2 x double> @test39(ptr %ptr) nounwind {
  %a = load double, ptr %ptr
  %v = insertelement <2 x double> undef, double %a, i32 0
  %shuffle = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %shuffle
  }

define <2 x double> @test40(ptr %ptr) nounwind {
  %v = load  <2 x double>,  ptr %ptr
  %shuffle = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %shuffle
  }

define <2 x double> @shuffle_v2f64_00(<2 x double> %a, <2 x double> %b) {
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0022(<4 x double> %a, <4 x double> %b) {
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x double> %shuffle
}

define <8 x i32> @ashr_v8i32(<8 x i32> %a, <8 x i32> %b) {
  %shift = ashr <8 x i32> %a, %b
  ret <8 x i32> %shift
}

define <8 x i32> @lshr_v8i32(<8 x i32> %a, <8 x i32> %b) {
  %shift = lshr <8 x i32> %a, %b
  ret <8 x i32> %shift
}

define <8 x i32> @shl_v8i32(<8 x i32> %a, <8 x i32> %b) {
  %shift = shl <8 x i32> %a, %b
  ret <8 x i32> %shift
}

define <8 x i32> @ashr_const_v8i32(<8 x i32> %a) {
  %shift = ashr <8 x i32> %a,  <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  ret <8 x i32> %shift
}

define <8 x i32> @lshr_const_v8i32(<8 x i32> %a) {
  %shift = lshr <8 x i32> %a,  <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  ret <8 x i32> %shift
}

define <8 x i32> @shl_const_v8i32(<8 x i32> %a) {
  %shift = shl <8 x i32> %a,  <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  ret <8 x i32> %shift
}

define <4 x i64> @ashr_v4i64(<4 x i64> %a, <4 x i64> %b) {
  %shift = ashr <4 x i64> %a, %b
  ret <4 x i64> %shift
}

define <4 x i64> @lshr_v4i64(<4 x i64> %a, <4 x i64> %b) {
  %shift = lshr <4 x i64> %a, %b
  ret <4 x i64> %shift
}

define <4 x i64> @shl_v4i64(<4 x i64> %a, <4 x i64> %b) {
  %shift = shl <4 x i64> %a, %b
  ret <4 x i64> %shift
}

define <4 x i64> @ashr_const_v4i64(<4 x i64> %a) {
  %shift = ashr <4 x i64> %a,  <i64 3, i64 3, i64 3, i64 3>
  ret <4 x i64> %shift
}

define <4 x i64> @lshr_const_v4i64(<4 x i64> %a) {
  %shift = lshr <4 x i64> %a,  <i64 3, i64 3, i64 3, i64 3>
  ret <4 x i64> %shift
}

define <4 x i64> @shl_const_v4i64(<4 x i64> %a) {
  %shift = shl <4 x i64> %a,  <i64 3, i64 3, i64 3, i64 3>
  ret <4 x i64> %shift
}

define <16 x i16> @ashr_v16i16(<16 x i16> %a, <16 x i16> %b) {
  %shift = ashr <16 x i16> %a, %b
  ret <16 x i16> %shift
}

define <16 x i16> @lshr_v16i16(<16 x i16> %a, <16 x i16> %b) {
  %shift = lshr <16 x i16> %a, %b
  ret <16 x i16> %shift
}

define <16 x i16> @shl_v16i16(<16 x i16> %a, <16 x i16> %b) {
  %shift = shl <16 x i16> %a, %b
  ret <16 x i16> %shift
}

define <16 x i16> @ashr_const_v16i16(<16 x i16> %a) {
  %shift = ashr <16 x i16> %a,  <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <16 x i16> %shift
}

define <16 x i16> @lshr_const_v16i16(<16 x i16> %a) {
  %shift = lshr <16 x i16> %a,  <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <16 x i16> %shift
}

define <16 x i16> @shl_const_v16i16(<16 x i16> %a) {
  %shift = shl <16 x i16> %a,  <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <16 x i16> %shift
}

define <4 x i32> @ashr_v4i32(<4 x i32> %a, <4 x i32> %b) {
  %shift = ashr <4 x i32> %a, %b
  ret <4 x i32> %shift
}

define <4 x i32> @shl_const_v4i32(<4 x i32> %a) {
  %shift = shl <4 x i32> %a,  <i32 3, i32 3, i32 3, i32 3>
  ret <4 x i32> %shift
}

define <2 x i64> @ashr_v2i64(<2 x i64> %a, <2 x i64> %b) {
  %shift = ashr <2 x i64> %a, %b
  ret <2 x i64> %shift
}

define <2 x i64> @shl_const_v2i64(<2 x i64> %a) {
  %shift = shl <2 x i64> %a,  <i64 3, i64 3>
  ret <2 x i64> %shift
}

define <8 x i16> @ashr_v8i16(<8 x i16> %a, <8 x i16> %b) {
  %shift = ashr <8 x i16> %a, %b
  ret <8 x i16> %shift
}

define <8 x i16> @lshr_v8i16(<8 x i16> %a, <8 x i16> %b) {
  %shift = lshr <8 x i16> %a, %b
  ret <8 x i16> %shift
}

define <8 x i16> @shl_v8i16(<8 x i16> %a, <8 x i16> %b) {
  %shift = shl <8 x i16> %a, %b
  ret <8 x i16> %shift
}

define <8 x i16> @ashr_const_v8i16(<8 x i16> %a) {
  %shift = ashr <8 x i16> %a,<i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <8 x i16> %shift
}

define <8 x i16> @lshr_const_v8i16(<8 x i16> %a) {
  %shift = lshr <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <8 x i16> %shift
}

define <8 x i16> @shl_const_v8i16(<8 x i16> %a) {
  %shift = shl <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <8 x i16> %shift
}

define <8 x i16> @zext_16i8_to_8i16(<16 x i8> %A) nounwind uwtable readnone ssp {
entry:
  %B = shufflevector <16 x i8> %A, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %C = zext <8 x i8> %B to <8 x i16>
  ret <8 x i16> %C
}

define   <32 x i8> @_broadcast32xi8(i8 %a) {
  %b = insertelement <32 x i8> undef, i8 %a, i32 0
  %c = shufflevector <32 x i8> %b, <32 x i8> undef, <32 x i32> zeroinitializer
  ret <32 x i8> %c
}

define   <16 x i8> @_broadcast16xi8(i8 %a) {
  %b = insertelement <16 x i8> undef, i8 %a, i32 0
  %c = shufflevector <16 x i8> %b, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %c
}

define   <16 x i16> @_broadcast16xi16(i16 %a) {
  %b = insertelement <16 x i16> undef, i16 %a, i32 0
  %c = shufflevector <16 x i16> %b, <16 x i16> undef, <16 x i32> zeroinitializer
  ret <16 x i16> %c
}

define   <8 x i16> @_broadcast8xi16(i16 %a) {
  %b = insertelement <8 x i16> undef, i16 %a, i32 0
  %c = shufflevector <8 x i16> %b, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %c
}

define <8 x i32> @_broadcast8xi32(i32 %a) {
  %b = insertelement <8 x i32> undef, i32 %a, i32 0
  %c = shufflevector <8 x i32> %b, <8 x i32> undef, <8 x i32> zeroinitializer
  ret <8 x i32> %c
}

define <4 x i32> @_broadcast4xi32(i32 %a) {
  %b = insertelement <4 x i32> undef, i32 %a, i32 0
  %c = shufflevector <4 x i32> %b, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %c
}

define <4 x i64> @_broadcast4xi64(i64 %a) {
  %b = insertelement <4 x i64> undef, i64 %a, i64 0
  %c = shufflevector <4 x i64> %b, <4 x i64> undef, <4 x i32> zeroinitializer
  ret <4 x i64> %c
}

define <2 x i64> @_broadcast2xi64(i64 %a) {
  %b = insertelement <2 x i64> undef, i64 %a, i64 0
  %c = shufflevector <2 x i64> %b, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %c
}

define   <8 x float> @_broadcast8xfloat(float %a) {
  %b = insertelement <8 x float> undef, float %a, i32 0
  %c = shufflevector <8 x float> %b, <8 x float> undef, <8 x i32> zeroinitializer
  ret <8 x float> %c
}

define   <4 x float> @_broadcast4xfloat(float %a) {
  %b = insertelement <4 x float> undef, float %a, i32 0
  %c = shufflevector <4 x float> %b, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %c
}

define   <4 x double> @_broadcast4xdouble(double %a) {
  %b = insertelement <4 x double> undef, double %a, i32 0
  %c = shufflevector <4 x double> %b, <4 x double> undef, <4 x i32> zeroinitializer
  ret <4 x double> %c
}

define   <2 x double> @_broadcast2xdouble(double %a) {
  %b = insertelement <2 x double> undef, double %a, i32 0
  %c = shufflevector <2 x double> %b, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %c
}

define <4 x float> @test_x86_fmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  %x = fmul <4 x float> %a0, %a1
  %res = fsub <4 x float> %x, %a2
  ret <4 x float> %res
}

define <32 x i8> @test_cmpgtb(<32 x i8> %A) {
; generate the follow code
;	 vpxor	 %ymm1, %ymm1, %ymm1
;  vpcmpgtb %ymm0, %ymm1, %ymm0
  %B = ashr <32 x i8> %A, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  ret <32 x i8> %B
}

define   <4 x float> @_inreg4xfloat(float %a) {
  %b = insertelement <4 x float> undef, float %a, i32 0
  %c = shufflevector <4 x float> %b, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %c
}

define   <8 x float> @_inreg8xfloat(float %a) {
  %b = insertelement <8 x float> undef, float %a, i32 0
  %c = shufflevector <8 x float> %b, <8 x float> undef, <8 x i32> zeroinitializer
  ret <8 x float> %c
}

define   <4 x double> @_inreg4xdouble(double %a) {
  %b = insertelement <4 x double> undef, double %a, i32 0
  %c = shufflevector <4 x double> %b, <4 x double> undef, <4 x i32> zeroinitializer
  ret <4 x double> %c
}

define <8 x i32> @test_mul_v8i32(<8 x i32> %arg1, <8 x i32> %arg2) #0 {
  %ret = mul <8 x i32> %arg1, %arg2
  ret <8 x i32> %ret
}
