; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s

define <4 x i32> @add_4i32(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @add_4i32
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   paddd %xmm1, %xmm0
  ;CHECK-NEXT:   retq
  %1 = add <4 x i32> %a0, <i32  1, i32 -2, i32  3, i32 -4>
  %2 = add <4 x i32> %a1, <i32 -1, i32  2, i32 -3, i32  4>
  %3 = add <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @add_4i32_commute(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @add_4i32_commute
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   paddd %xmm1, %xmm0
  ;CHECK-NEXT:   retq
  %1 = add <4 x i32> <i32  1, i32 -2, i32  3, i32 -4>, %a0
  %2 = add <4 x i32> <i32 -1, i32  2, i32 -3, i32  4>, %a1
  %3 = add <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @mul_4i32(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @mul_4i32
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   pmulld %xmm1, %xmm0
  ;CHECK-NEXT:   pmulld .LCPI2_0(%rip), %xmm0
  ;CHECK-NEXT:   retq
  %1 = mul <4 x i32> %a0, <i32 1, i32 2, i32 3, i32 4>
  %2 = mul <4 x i32> %a1, <i32 4, i32 3, i32 2, i32 1>
  %3 = mul <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @mul_4i32_commute(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @mul_4i32_commute
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   pmulld %xmm1, %xmm0
  ;CHECK-NEXT:   pmulld .LCPI3_0(%rip), %xmm0
  ;CHECK-NEXT:   retq
  %1 = mul <4 x i32> <i32 1, i32 2, i32 3, i32 4>, %a0
  %2 = mul <4 x i32> <i32 4, i32 3, i32 2, i32 1>, %a1
  %3 = mul <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @and_4i32(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @and_4i32
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   andps %xmm1, %xmm0
  ;CHECK-NEXT:   andps .LCPI4_0(%rip), %xmm0
  ;CHECK-NEXT:   retq
  %1 = and <4 x i32> %a0, <i32 -2, i32 -2, i32  3, i32  3>
  %2 = and <4 x i32> %a1, <i32 -1, i32 -1, i32  1, i32  1>
  %3 = and <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @and_4i32_commute(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @and_4i32_commute
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   andps %xmm1, %xmm0
  ;CHECK-NEXT:   andps .LCPI5_0(%rip), %xmm0
  ;CHECK-NEXT:   retq
  %1 = and <4 x i32> <i32 -2, i32 -2, i32  3, i32  3>, %a0
  %2 = and <4 x i32> <i32 -1, i32 -1, i32  1, i32  1>, %a1
  %3 = and <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @or_4i32(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @or_4i32
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   orps %xmm1, %xmm0
  ;CHECK-NEXT:   orps .LCPI6_0(%rip), %xmm0
  ;CHECK-NEXT:   retq
  %1 = or <4 x i32> %a0, <i32 -2, i32 -2, i32  3, i32  3>
  %2 = or <4 x i32> %a1, <i32 -1, i32 -1, i32  1, i32  1>
  %3 = or <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @or_4i32_commute(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @or_4i32_commute
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   orps %xmm1, %xmm0
  ;CHECK-NEXT:   orps .LCPI7_0(%rip), %xmm0
  ;CHECK-NEXT:   retq
  %1 = or <4 x i32> <i32 -2, i32 -2, i32  3, i32  3>, %a0 
  %2 = or <4 x i32> <i32 -1, i32 -1, i32  1, i32  1>, %a1
  %3 = or <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @xor_4i32(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @xor_4i32
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   xorps %xmm1, %xmm0
  ;CHECK-NEXT:   xorps .LCPI8_0(%rip), %xmm0
  ;CHECK-NEXT:   retq
  %1 = xor <4 x i32> %a0, <i32 -2, i32 -2, i32  3, i32  3>
  %2 = xor <4 x i32> %a1, <i32 -1, i32 -1, i32  1, i32  1>
  %3 = xor <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @xor_4i32_commute(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL:  @xor_4i32_commute
  ;CHECK:        # BB#0:
  ;CHECK-NEXT:   xorps %xmm1, %xmm0
  ;CHECK-NEXT:   xorps .LCPI9_0(%rip), %xmm0
  ;CHECK-NEXT:   retq
  %1 = xor <4 x i32> <i32 -2, i32 -2, i32  3, i32  3>, %a0
  %2 = xor <4 x i32> <i32 -1, i32 -1, i32  1, i32  1>, %a1
  %3 = xor <4 x i32> %1, %2
  ret <4 x i32> %3
}
