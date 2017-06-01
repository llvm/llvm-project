; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=sse2 | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=sse2 | FileCheck %s --check-prefix=X32
; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=avx2 | FileCheck %s --check-prefix=X32AVX

; Use movq or movsd to load / store i64 values if sse2 is available.
; rdar://6659858

define void @foo(i64* %x, i64* %y) {
; X64-LABEL: foo:
; X64:       # BB#0:
; X64-NEXT:    movq (%rsi), %rax
; X64-NEXT:    movq %rax, (%rdi)
; X64-NEXT:    retq
;
; X32-LABEL: foo:
; X32:       # BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X32-NEXT:    movsd {{.*#+}} xmm0 = mem[0],zero
; X32-NEXT:    movsd %xmm0, (%eax)
; X32-NEXT:    retl
  %tmp1 = load i64, i64* %y, align 8
  store i64 %tmp1, i64* %x, align 8
  ret void
}

; Verify that a 64-bit chunk extracted from a vector is stored with a movq
; regardless of whether the system is 64-bit.

define void @store_i64_from_vector(<8 x i16> %x, <8 x i16> %y, i64* %i) {
; X64-LABEL: store_i64_from_vector:
; X64:       # BB#0:
; X64-NEXT:    paddw %xmm1, %xmm0
; X64-NEXT:    movq %xmm0, (%rdi)
; X64-NEXT:    retq
;
; X32-LABEL: store_i64_from_vector:
; X32:       # BB#0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    paddw %xmm1, %xmm0
; X32-NEXT:    movq %xmm0, (%eax)
; X32-NEXT:    retl
  %z = add <8 x i16> %x, %y                          ; force execution domain
  %bc = bitcast <8 x i16> %z to <2 x i64>
  %vecext = extractelement <2 x i64> %bc, i32 0
  store i64 %vecext, i64* %i, align 8
  ret void
}

define void @store_i64_from_vector256(<16 x i16> %x, <16 x i16> %y, i64* %i) {
; X32AVX-LABEL: store_i64_from_vector256:
; X32AVX:       # BB#0:
; X32AVX-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32AVX-NEXT:    vpaddw %ymm1, %ymm0, %ymm0
; X32AVX-NEXT:    vextracti128 $1, %ymm0, %xmm0
; X32AVX-NEXT:    vmovq %xmm0, (%eax)
; X32AVX-NEXT:    vzeroupper
; X32AVX-NEXT:    retl
  %z = add <16 x i16> %x, %y                          ; force execution domain
  %bc = bitcast <16 x i16> %z to <4 x i64>
  %vecext = extractelement <4 x i64> %bc, i32 2
  store i64 %vecext, i64* %i, align 8
  ret void
}

; PR23476
; Handle extraction from a non-simple / pre-legalization type.

define void @PR23476(<5 x i64> %in, i64* %out, i32 %index) {
; X32-LABEL: PR23476:
; X32: andl $7, %eax
; X32:         movsd {{.*#+}} xmm0 = mem[0],zero
; X32:         movsd {{.*#+}} xmm0 = mem[0],zero
; X32-NEXT:    movsd %xmm0, (%ecx)
  %ext = extractelement <5 x i64> %in, i32 %index
  store i64 %ext, i64* %out, align 8
  ret void
}
