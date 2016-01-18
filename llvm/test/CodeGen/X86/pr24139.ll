; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s

; Check that we do not get excessive spilling from splitting of constant live ranges.

; CHECK-LABEL: PR24139:
; CHECK: # 16-byte Spill
; CHECK-NOT: # 16-byte Spill
; CHECK: retq

define <2 x double> @PR24139(<2 x double> %arg, <2 x double> %arg1, <2 x double> %arg2) {
  %tmp = bitcast <2 x double> %arg to <4 x float>
  %tmp3 = fmul <4 x float> %tmp, <float 0x3FE45F3060000000, float 0x3FE45F3060000000, float 0x3FE45F3060000000, float 0x3FE45F3060000000>
  %tmp4 = bitcast <2 x double> %arg to <4 x i32>
  %tmp5 = and <4 x i32> %tmp4, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %tmp6 = or <4 x i32> %tmp5, <i32 1056964608, i32 1056964608, i32 1056964608, i32 1056964608>
  %tmp7 = bitcast <4 x i32> %tmp6 to <4 x float>
  %tmp8 = fadd <4 x float> %tmp3, %tmp7
  %tmp9 = tail call <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float> %tmp8) #2
  %tmp10 = bitcast <4 x i32> %tmp9 to <2 x i64>
  %tmp11 = tail call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %tmp9) #2
  %tmp12 = fmul <4 x float> %tmp11, <float 0x3FF921FB40000000, float 0x3FF921FB40000000, float 0x3FF921FB40000000, float 0x3FF921FB40000000>
  %tmp13 = fsub <4 x float> %tmp, %tmp12
  %tmp14 = fmul <4 x float> %tmp11, <float 0x3E74442D00000000, float 0x3E74442D00000000, float 0x3E74442D00000000, float 0x3E74442D00000000>
  %tmp15 = fsub <4 x float> %tmp13, %tmp14
  %tmp16 = fmul <4 x float> %tmp15, %tmp15
  %tmp17 = fmul <4 x float> %tmp15, %tmp16
  %tmp18 = fmul <4 x float> %tmp16, <float 0xBF56493260000000, float 0xBF56493260000000, float 0xBF56493260000000, float 0xBF56493260000000>
  %tmp19 = fadd <4 x float> %tmp18, <float 0x3FA55406C0000000, float 0x3FA55406C0000000, float 0x3FA55406C0000000, float 0x3FA55406C0000000>
  %tmp20 = fmul <4 x float> %tmp16, <float 0xBF29918DC0000000, float 0xBF29918DC0000000, float 0xBF29918DC0000000, float 0xBF29918DC0000000>
  %tmp21 = fadd <4 x float> %tmp20, <float 0x3F81106840000000, float 0x3F81106840000000, float 0x3F81106840000000, float 0x3F81106840000000>
  %tmp22 = fmul <4 x float> %tmp16, %tmp19
  %tmp23 = fadd <4 x float> %tmp22, <float 0xBFDFFFFBE0000000, float 0xBFDFFFFBE0000000, float 0xBFDFFFFBE0000000, float 0xBFDFFFFBE0000000>
  %tmp24 = fmul <4 x float> %tmp16, %tmp21
  %tmp25 = fadd <4 x float> %tmp24, <float 0xBFC5555420000000, float 0xBFC5555420000000, float 0xBFC5555420000000, float 0xBFC5555420000000>
  %tmp26 = fmul <4 x float> %tmp16, %tmp23
  %tmp27 = fadd <4 x float> %tmp26, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %tmp28 = fmul <4 x float> %tmp17, %tmp25
  %tmp29 = fadd <4 x float> %tmp15, %tmp28
  %tmp30 = and <2 x i64> %tmp10, <i64 4294967297, i64 4294967297>
  %tmp31 = bitcast <2 x i64> %tmp30 to <4 x i32>
  %tmp32 = icmp eq <4 x i32> %tmp31, zeroinitializer
  %tmp33 = sext <4 x i1> %tmp32 to <4 x i32>
  %tmp34 = bitcast <4 x i32> %tmp33 to <4 x float>
  %tmp35 = tail call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %tmp27, <4 x float> %tmp29, <4 x float> %tmp34) #2
  %tmp36 = and <2 x i64> %tmp10, <i64 8589934594, i64 8589934594>
  %tmp37 = bitcast <2 x i64> %tmp36 to <4 x i32>
  %tmp38 = icmp eq <4 x i32> %tmp37, zeroinitializer
  %tmp39 = sext <4 x i1> %tmp38 to <4 x i32>
  %tmp40 = bitcast <4 x float> %tmp35 to <4 x i32>
  %tmp41 = xor <4 x i32> %tmp40, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %tmp42 = bitcast <4 x i32> %tmp41 to <4 x float>
  %tmp43 = bitcast <4 x i32> %tmp39 to <4 x float>
  %tmp44 = tail call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %tmp42, <4 x float> %tmp35, <4 x float> %tmp43) #2
  %tmp45 = bitcast <2 x double> %arg1 to <4 x float>
  %tmp46 = fmul <4 x float> %tmp45, <float 0x3FE45F3060000000, float 0x3FE45F3060000000, float 0x3FE45F3060000000, float 0x3FE45F3060000000>
  %tmp47 = bitcast <2 x double> %arg1 to <4 x i32>
  %tmp48 = and <4 x i32> %tmp47, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %tmp49 = or <4 x i32> %tmp48, <i32 1056964608, i32 1056964608, i32 1056964608, i32 1056964608>
  %tmp50 = bitcast <4 x i32> %tmp49 to <4 x float>
  %tmp51 = fadd <4 x float> %tmp46, %tmp50
  %tmp52 = tail call <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float> %tmp51) #2
  %tmp53 = bitcast <4 x i32> %tmp52 to <2 x i64>
  %tmp54 = tail call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %tmp52) #2
  %tmp55 = fmul <4 x float> %tmp54, <float 0x3FF921FB40000000, float 0x3FF921FB40000000, float 0x3FF921FB40000000, float 0x3FF921FB40000000>
  %tmp56 = fsub <4 x float> %tmp45, %tmp55
  %tmp57 = fmul <4 x float> %tmp54, <float 0x3E74442D00000000, float 0x3E74442D00000000, float 0x3E74442D00000000, float 0x3E74442D00000000>
  %tmp58 = fsub <4 x float> %tmp56, %tmp57
  %tmp59 = fmul <4 x float> %tmp58, %tmp58
  %tmp60 = fmul <4 x float> %tmp58, %tmp59
  %tmp61 = fmul <4 x float> %tmp59, <float 0xBF56493260000000, float 0xBF56493260000000, float 0xBF56493260000000, float 0xBF56493260000000>
  %tmp62 = fadd <4 x float> %tmp61, <float 0x3FA55406C0000000, float 0x3FA55406C0000000, float 0x3FA55406C0000000, float 0x3FA55406C0000000>
  %tmp63 = fmul <4 x float> %tmp59, <float 0xBF29918DC0000000, float 0xBF29918DC0000000, float 0xBF29918DC0000000, float 0xBF29918DC0000000>
  %tmp64 = fadd <4 x float> %tmp63, <float 0x3F81106840000000, float 0x3F81106840000000, float 0x3F81106840000000, float 0x3F81106840000000>
  %tmp65 = fmul <4 x float> %tmp59, %tmp62
  %tmp66 = fadd <4 x float> %tmp65, <float 0xBFDFFFFBE0000000, float 0xBFDFFFFBE0000000, float 0xBFDFFFFBE0000000, float 0xBFDFFFFBE0000000>
  %tmp67 = fmul <4 x float> %tmp59, %tmp64
  %tmp68 = fadd <4 x float> %tmp67, <float 0xBFC5555420000000, float 0xBFC5555420000000, float 0xBFC5555420000000, float 0xBFC5555420000000>
  %tmp69 = fmul <4 x float> %tmp59, %tmp66
  %tmp70 = fadd <4 x float> %tmp69, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %tmp71 = fmul <4 x float> %tmp60, %tmp68
  %tmp72 = fadd <4 x float> %tmp58, %tmp71
  %tmp73 = and <2 x i64> %tmp53, <i64 4294967297, i64 4294967297>
  %tmp74 = bitcast <2 x i64> %tmp73 to <4 x i32>
  %tmp75 = icmp eq <4 x i32> %tmp74, zeroinitializer
  %tmp76 = sext <4 x i1> %tmp75 to <4 x i32>
  %tmp77 = bitcast <4 x i32> %tmp76 to <4 x float>
  %tmp78 = tail call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %tmp70, <4 x float> %tmp72, <4 x float> %tmp77) #2
  %tmp79 = and <2 x i64> %tmp53, <i64 8589934594, i64 8589934594>
  %tmp80 = bitcast <2 x i64> %tmp79 to <4 x i32>
  %tmp81 = icmp eq <4 x i32> %tmp80, zeroinitializer
  %tmp82 = sext <4 x i1> %tmp81 to <4 x i32>
  %tmp83 = bitcast <4 x float> %tmp78 to <4 x i32>
  %tmp84 = xor <4 x i32> %tmp83, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %tmp85 = bitcast <4 x i32> %tmp84 to <4 x float>
  %tmp86 = bitcast <4 x i32> %tmp82 to <4 x float>
  %tmp87 = tail call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %tmp85, <4 x float> %tmp78, <4 x float> %tmp86) #2
  %tmp88 = fadd <4 x float> %tmp44, %tmp87
  %tmp89 = bitcast <2 x double> %arg2 to <4 x float>
  %tmp90 = fmul <4 x float> %tmp89, <float 0x3FE45F3060000000, float 0x3FE45F3060000000, float 0x3FE45F3060000000, float 0x3FE45F3060000000>
  %tmp91 = bitcast <2 x double> %arg2 to <4 x i32>
  %tmp92 = and <4 x i32> %tmp91, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %tmp93 = or <4 x i32> %tmp92, <i32 1056964608, i32 1056964608, i32 1056964608, i32 1056964608>
  %tmp94 = bitcast <4 x i32> %tmp93 to <4 x float>
  %tmp95 = fadd <4 x float> %tmp90, %tmp94
  %tmp96 = tail call <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float> %tmp95) #2
  %tmp97 = bitcast <4 x i32> %tmp96 to <2 x i64>
  %tmp98 = tail call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %tmp96) #2
  %tmp99 = fmul <4 x float> %tmp98, <float 0x3FF921FB40000000, float 0x3FF921FB40000000, float 0x3FF921FB40000000, float 0x3FF921FB40000000>
  %tmp100 = fsub <4 x float> %tmp89, %tmp99
  %tmp101 = fmul <4 x float> %tmp98, <float 0x3E74442D00000000, float 0x3E74442D00000000, float 0x3E74442D00000000, float 0x3E74442D00000000>
  %tmp102 = fsub <4 x float> %tmp100, %tmp101
  %tmp103 = fmul <4 x float> %tmp102, %tmp102
  %tmp104 = fmul <4 x float> %tmp102, %tmp103
  %tmp105 = fmul <4 x float> %tmp103, <float 0xBF56493260000000, float 0xBF56493260000000, float 0xBF56493260000000, float 0xBF56493260000000>
  %tmp106 = fadd <4 x float> %tmp105, <float 0x3FA55406C0000000, float 0x3FA55406C0000000, float 0x3FA55406C0000000, float 0x3FA55406C0000000>
  %tmp107 = fmul <4 x float> %tmp103, <float 0xBF29918DC0000000, float 0xBF29918DC0000000, float 0xBF29918DC0000000, float 0xBF29918DC0000000>
  %tmp108 = fadd <4 x float> %tmp107, <float 0x3F81106840000000, float 0x3F81106840000000, float 0x3F81106840000000, float 0x3F81106840000000>
  %tmp109 = fmul <4 x float> %tmp103, %tmp106
  %tmp110 = fadd <4 x float> %tmp109, <float 0xBFDFFFFBE0000000, float 0xBFDFFFFBE0000000, float 0xBFDFFFFBE0000000, float 0xBFDFFFFBE0000000>
  %tmp111 = fmul <4 x float> %tmp103, %tmp108
  %tmp112 = fadd <4 x float> %tmp111, <float 0xBFC5555420000000, float 0xBFC5555420000000, float 0xBFC5555420000000, float 0xBFC5555420000000>
  %tmp113 = fmul <4 x float> %tmp103, %tmp110
  %tmp114 = fadd <4 x float> %tmp113, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %tmp115 = fmul <4 x float> %tmp104, %tmp112
  %tmp116 = fadd <4 x float> %tmp102, %tmp115
  %tmp117 = and <2 x i64> %tmp97, <i64 4294967297, i64 4294967297>
  %tmp118 = bitcast <2 x i64> %tmp117 to <4 x i32>
  %tmp119 = icmp eq <4 x i32> %tmp118, zeroinitializer
  %tmp120 = sext <4 x i1> %tmp119 to <4 x i32>
  %tmp121 = bitcast <4 x i32> %tmp120 to <4 x float>
  %tmp122 = tail call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %tmp114, <4 x float> %tmp116, <4 x float> %tmp121) #2
  %tmp123 = and <2 x i64> %tmp97, <i64 8589934594, i64 8589934594>
  %tmp124 = bitcast <2 x i64> %tmp123 to <4 x i32>
  %tmp125 = icmp eq <4 x i32> %tmp124, zeroinitializer
  %tmp126 = sext <4 x i1> %tmp125 to <4 x i32>
  %tmp127 = bitcast <4 x float> %tmp122 to <4 x i32>
  %tmp128 = xor <4 x i32> %tmp127, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %tmp129 = bitcast <4 x i32> %tmp128 to <4 x float>
  %tmp130 = bitcast <4 x i32> %tmp126 to <4 x float>
  %tmp131 = tail call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %tmp129, <4 x float> %tmp122, <4 x float> %tmp130) #2
  %tmp132 = fadd <4 x float> %tmp88, %tmp131
  %tmp133 = bitcast <4 x float> %tmp132 to <2 x double>
  ret <2 x double> %tmp133
}

declare <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float>)
declare <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32>)
declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>)
