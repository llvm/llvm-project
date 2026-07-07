; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512bw,+avx512vl,+avx512dq | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512f | FileCheck %s --check-prefix=AVX512F

define i32 @test_v8i1_scalar_mask(<8 x i32> %a, <8 x i32> %b, i8 %mask) {
; CHECK-LABEL: test_v8i1_scalar_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vpcmpneqd %ymm1, %ymm0, %k0 {%k1}
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    kortestb %k0, %k0
; CHECK-NEXT:    sete %al
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %cmp = icmp ne <8 x i32> %a, %b
  %bc = bitcast <8 x i1> %cmp to i8
  %and = and i8 %bc, %mask
  %eq = icmp eq i8 %and, 0
  %ret = zext i1 %eq to i32
  ret i32 %ret
}

define i32 @test_v16i1_scalar_mask(<2 x i64> %a, <2 x i64> %b, i16 %mask) {
; CHECK-LABEL: test_v16i1_scalar_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vpcmpneqb %xmm1, %xmm0, %k0 {%k1}
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    kortestw %k0, %k0
; CHECK-NEXT:    sete %al
; CHECK-NEXT:    retq
  %va = bitcast <2 x i64> %a to <16 x i8>
  %vb = bitcast <2 x i64> %b to <16 x i8>
  %cmp = icmp ne <16 x i8> %va, %vb
  %bc = bitcast <16 x i1> %cmp to i16
  %and = and i16 %bc, %mask
  %eq = icmp eq i16 %and, 0
  %ret = zext i1 %eq to i32
  ret i32 %ret
}

define i32 @test_v16i1_scalar_mask_ne(<2 x i64> %a, <2 x i64> %b,
                                      i16 %mask) {
; CHECK-LABEL: test_v16i1_scalar_mask_ne:
; CHECK:       # %bb.0:
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vpcmpneqb %xmm1, %xmm0, %k0 {%k1}
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    kortestw %k0, %k0
; CHECK-NEXT:    setne %al
; CHECK-NEXT:    retq
  %va = bitcast <2 x i64> %a to <16 x i8>
  %vb = bitcast <2 x i64> %b to <16 x i8>
  %cmp = icmp ne <16 x i8> %va, %vb
  %bc = bitcast <16 x i1> %cmp to i16
  %and = and i16 %bc, %mask
  %ne = icmp ne i16 %and, 0
  %ret = zext i1 %ne to i32
  ret i32 %ret
}

define i32 @test_v32i1_scalar_mask(<32 x i16> %a, <32 x i16> %b, i32 %mask) {
; CHECK-LABEL: test_v32i1_scalar_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    kmovd %edi, %k1
; CHECK-NEXT:    vpcmpneqw %zmm1, %zmm0, %k0 {%k1}
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    kortestd %k0, %k0
; CHECK-NEXT:    sete %al
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  %cmp = icmp ne <32 x i16> %a, %b
  %bc = bitcast <32 x i1> %cmp to i32
  %and = and i32 %bc, %mask
  %eq = icmp eq i32 %and, 0
  %ret = zext i1 %eq to i32
  ret i32 %ret
}

; Keep constant masks on the scalar immediate-test path for now. Materializing
; a one-use immediate mask in a k-register is not clearly better.
define i32 @test_v16i1_constant_mask(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_v16i1_constant_mask:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vpcmpneqb %xmm1, %xmm0, %k0
; CHECK-NEXT:    kmovd %k0, %ecx
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    testb $127, %cl
; CHECK-NEXT:    sete %al
; CHECK-NEXT:    retq
  %va = bitcast <2 x i64> %a to <16 x i8>
  %vb = bitcast <2 x i64> %b to <16 x i8>
  %cmp = icmp ne <16 x i8> %va, %vb
  %bc = bitcast <16 x i1> %cmp to i16
  %and = and i16 %bc, 127
  %eq = icmp eq i16 %and, 0
  %ret = zext i1 %eq to i32
  ret i32 %ret
}

define i32 @test_v16i1_scalar_mask_avx512f(<16 x i32> %a, <16 x i32> %b,
                                           i16 %mask) {
; AVX512F-LABEL: test_v16i1_scalar_mask_avx512f:
; AVX512F:       # %bb.0:
; AVX512F-NEXT:    kmovw %edi, %k1
; AVX512F-NEXT:    vpcmpneqd %zmm1, %zmm0, %k0 {%k1}
; AVX512F-NEXT:    xorl %eax, %eax
; AVX512F-NEXT:    kortestw %k0, %k0
; AVX512F-NEXT:    sete %al
; AVX512F-NEXT:    vzeroupper
; AVX512F-NEXT:    retq
  %cmp = icmp ne <16 x i32> %a, %b
  %bc = bitcast <16 x i1> %cmp to i16
  %and = and i16 %bc, %mask
  %eq = icmp eq i16 %and, 0
  %ret = zext i1 %eq to i32
  ret i32 %ret
}
