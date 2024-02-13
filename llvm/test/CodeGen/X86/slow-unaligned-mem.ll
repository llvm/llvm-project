; Intel chips with slow unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium3      2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SCALAR
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium3m     2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SCALAR
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium-m     2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium4      2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium4m     2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=yonah         2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=prescott      2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=nocona        2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=core2         2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=penryn        2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bonnell       2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE

; Intel chips with fast unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=silvermont     2>&1 | FileCheck %s --check-prefixes=FAST,FAST-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=nehalem        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=westmere       2>&1 | FileCheck %s --check-prefixes=FAST,FAST-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=sandybridge    2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX128
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=ivybridge      2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX128
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=haswell        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=broadwell      2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=knl            2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX512
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=skylake-avx512 2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256

; AMD chips with slow unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon-4      2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SCALAR
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon-xp     2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SCALAR
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=k8            2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=opteron       2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon64      2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon-fx     2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=k8-sse3       2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=opteron-sse3  2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon64-sse3 2>&1 | FileCheck %s --check-prefixes=SLOW,SLOW-SSE

; AMD chips with fast unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=amdfam10      2>&1 | FileCheck %s --check-prefixes=FAST,FAST-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=barcelona     2>&1 | FileCheck %s --check-prefixes=FAST,FAST-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=btver1        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=btver2        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bdver1        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bdver2        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bdver3        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bdver4        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=znver1        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=znver2        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=znver3        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX256
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=znver4        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-AVX512

; Other chips with slow unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=c3-2          2>&1 | FileCheck %s --check-prefixes=SLOW

; Verify that the slow/fast unaligned memory attribute is set correctly for each CPU model.
; Slow chips use 4-byte stores. Fast chips with SSE or later use something other than 4-byte stores.
; Chips that don't have SSE use 4-byte stores either way, so they're not tested.

; Also verify that SSE4.2 or SSE4a imply fast unaligned accesses.

; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=sse4.2       2>&1 | FileCheck %s --check-prefixes=FAST,FAST-SSE
; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=sse4a        2>&1 | FileCheck %s --check-prefixes=FAST,FAST-SSE

; SLOW-NOT: not a recognized processor
; FAST-NOT: not a recognized processor
define void @store_zeros(ptr %a) {
; SLOW-SCALAR-LABEL: store_zeros:
; SLOW-SCALAR:       # %bb.0:
; SLOW-SCALAR-NEXT:    movl {{[0-9]+}}(%esp), %eax
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NEXT:    movl $0
; SLOW-SCALAR-NOT:     movl
;
; SLOW-SSE-LABEL: store_zeros:
; SLOW-SSE:       # %bb.0:
; SLOW-SSE-NEXT:    movl {{[0-9]+}}(%esp), %eax
; SLOW-SSE-NEXT:    xorps %xmm0, %xmm0
; SLOW-SSE-NEXT:    movsd %xmm0
; SLOW-SSE-NEXT:    movsd %xmm0
; SLOW-SSE-NEXT:    movsd %xmm0
; SLOW-SSE-NEXT:    movsd %xmm0
; SLOW-SSE-NEXT:    movsd %xmm0
; SLOW-SSE-NEXT:    movsd %xmm0
; SLOW-SSE-NEXT:    movsd %xmm0
; SLOW-SSE-NEXT:    movsd %xmm0
; SLOW-SSE-NOT:     movsd
;
; FAST-SSE-LABEL: store_zeros:
; FAST-SSE:       # %bb.0:
; FAST-SSE-NEXT:    movl {{[0-9]+}}(%esp), %eax
; FAST-SSE-NEXT:    xorps %xmm0, %xmm0
; FAST-SSE-NEXT:    movups %xmm0
; FAST-SSE-NEXT:    movups %xmm0
; FAST-SSE-NEXT:    movups %xmm0
; FAST-SSE-NEXT:    movups %xmm0
; FAST-SSE-NOT:     movups
;
; FAST-AVX128-LABEL: store_zeros:
; FAST-AVX128:       # %bb.0:
; FAST-AVX128-NEXT:    movl {{[0-9]+}}(%esp), %eax
; FAST-AVX128-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; FAST-AVX128-NEXT:    vmovups %xmm0
; FAST-AVX128-NEXT:    vmovups %xmm0
; FAST-AVX128-NEXT:    vmovups %xmm0
; FAST-AVX128-NEXT:    vmovups %xmm0
; FAST-AVX128-NOT:     vmovups
;
; FAST-AVX256-LABEL: store_zeros:
; FAST-AVX256:       # %bb.0:
; FAST-AVX256-NEXT:    movl {{[0-9]+}}(%esp), %eax
; FAST-AVX256-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; FAST-AVX256-NEXT:    vmovups %ymm0
; FAST-AVX256-NEXT:    vmovups %ymm0
; FAST-AVX256-NOT:     vmovups
;
; FAST-AVX512-LABEL: store_zeros:
; FAST-AVX512:       # %bb.0:
; FAST-AVX512-NEXT:    movl {{[0-9]+}}(%esp), %eax
; FAST-AVX512-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; FAST-AVX512-NEXT:    vmovups %zmm0, (%eax)
; FAST-AVX512-NOT:     vmovups
  call void @llvm.memset.p0.i64(ptr %a, i8 0, i64 64, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)

