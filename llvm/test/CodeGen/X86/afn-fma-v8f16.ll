; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=alderlake | FileCheck %s --check-prefix=FMA
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=ivybridge | FileCheck %s --check-prefix=NOFMA
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=sapphirerapids | FileCheck %s --check-prefix=FP16

; Check v8f16 FMA lowering with approximate, exact, and native FP16 paths.

define <8 x half> @afn_fma(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; FMA-LABEL: afn_fma:
; FMA:       # %bb.0:
; FMA-NEXT:    vcvtph2ps %xmm2, %ymm2
; FMA-NEXT:    vcvtph2ps %xmm0, %ymm0
; FMA-NEXT:    vcvtph2ps %xmm1, %ymm1
; FMA-NEXT:    vfmadd213ps {{.*#+}} ymm1 = (ymm0 * ymm1) + ymm2
; FMA-NEXT:    vcvtps2ph $4, %ymm1, %xmm0
; FMA-NEXT:    vzeroupper
; FMA-NEXT:    retq
;
; NOFMA-LABEL: afn_fma:
; NOFMA:         callq fma@PLT
; NOFMA-NEXT:    callq __truncdfhf2@PLT
; NOFMA:         retq
;
; FP16-LABEL: afn_fma:
; FP16:         vfmadd213ph %xmm2, %xmm1, %xmm0
; FP16-NEXT:    retq
  %r = call afn <8 x half> @llvm.fma.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c)
  ret <8 x half> %r
}

define <8 x half> @exact_fma(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; FMA-LABEL: exact_fma:
; FMA:         vfmadd213sd
; FMA-COUNT-8: callq __truncdfhf2@PLT
; FMA:         retq
;
; NOFMA-LABEL: exact_fma:
; NOFMA:         callq fma@PLT
; NOFMA-NEXT:    callq __truncdfhf2@PLT
; NOFMA:         retq
;
; FP16-LABEL: exact_fma:
; FP16:         vfmadd213ph %xmm2, %xmm1, %xmm0
; FP16-NEXT:    retq
  %r = call <8 x half> @llvm.fma.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c)
  ret <8 x half> %r
}

define <8 x half> @contract_fma(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; FMA-LABEL: contract_fma:
; FMA:         vfmadd213sd
; FMA-COUNT-8: callq __truncdfhf2@PLT
; FMA:         retq
;
; NOFMA-LABEL: contract_fma:
; NOFMA:         callq fma@PLT
; NOFMA-NEXT:    callq __truncdfhf2@PLT
; NOFMA:         retq
;
; FP16-LABEL: contract_fma:
; FP16:         vfmadd213ph %xmm2, %xmm1, %xmm0
; FP16-NEXT:    retq
  %r = call contract <8 x half> @llvm.fma.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c)
  ret <8 x half> %r
}

define <8 x half> @fast_fma(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; FMA-LABEL: fast_fma:
; FMA:       # %bb.0:
; FMA-NEXT:    vcvtph2ps %xmm2, %ymm2
; FMA-NEXT:    vcvtph2ps %xmm0, %ymm0
; FMA-NEXT:    vcvtph2ps %xmm1, %ymm1
; FMA-NEXT:    vfmadd213ps {{.*#+}} ymm1 = (ymm0 * ymm1) + ymm2
; FMA-NEXT:    vcvtps2ph $4, %ymm1, %xmm0
; FMA-NEXT:    vzeroupper
; FMA-NEXT:    retq
;
; NOFMA-LABEL: fast_fma:
; NOFMA:       # %bb.0:
; NOFMA-NEXT:    vcvtph2ps %xmm1, %ymm1
; NOFMA-NEXT:    vcvtph2ps %xmm0, %ymm0
; NOFMA-NEXT:    vmulps %ymm1, %ymm0, %ymm0
; NOFMA-NEXT:    vcvtps2ph $4, %ymm0, %xmm0
; NOFMA-NEXT:    vcvtph2ps %xmm0, %ymm0
; NOFMA-NEXT:    vcvtph2ps %xmm2, %ymm1
; NOFMA-NEXT:    vaddps %ymm1, %ymm0, %ymm0
; NOFMA-NEXT:    vcvtps2ph $4, %ymm0, %xmm0
; NOFMA-NEXT:    vzeroupper
; NOFMA-NEXT:    retq
;
; FP16-LABEL: fast_fma:
; FP16:         vfmadd213ph %xmm2, %xmm1, %xmm0
; FP16-NEXT:    retq
  %r = call fast <8 x half> @llvm.fma.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c)
  ret <8 x half> %r
}

declare <8 x half> @llvm.fma.v8f16(<8 x half>, <8 x half>, <8 x half>)
