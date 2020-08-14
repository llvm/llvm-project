; RUN: llc -aarch64-sve-vector-bits-min=128  -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=384  -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 -aarch64-enable-atomic-cfg-tidy=false < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 -aarch64-enable-atomic-cfg-tidy=false < %s 2>%t | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

; Test we can code generater patterns of the form:
;   fixed_length_vector = ISD::EXTRACT_SUBVECTOR scalable_vector, 0
;   scalable_vector = ISD::INSERT_SUBVECTOR scalable_vector, fixed_length_vector, 0
;
; NOTE: Currently shufflevector does not support scalable vectors so it cannot
; be used to model the above operations.  Instead these tests rely on knowing
; how fixed length operation are lowered to scalable ones, with multiple blocks
; ensuring insert/extract sequences are not folded away.

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

define void @subvector_v8i32(<8 x i32> *%in, <8 x i32>* %out) #0 {
; CHECK-LABEL: subvector_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK: ld1w { [[DATA:z[0-9]+.s]] }, [[PG]]/z, [x0]
; CHECK: st1w { [[DATA]] }, [[PG]], [x1]
; CHECK: ret
  %a = load <8 x i32>, <8 x i32>* %in
  br label %bb1

bb1:
  store <8 x i32> %a, <8 x i32>* %out
  ret void
}

define void @subvector_v16i32(<16 x i32> *%in, <16 x i32>* %out) #0 {
; CHECK-LABEL: subvector_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512: ld1w { [[DATA:z[0-9]+.s]] }, [[PG]]/z, [x0]
; VBITS_GE_512: st1w { [[DATA]] }, [[PG]], [x1]
; CHECKT: ret
  %a = load <16 x i32>, <16 x i32>* %in
  br label %bb1

bb1:
  store <16 x i32> %a, <16 x i32>* %out
  ret void
}

define void @subvector_v32i32(<32 x i32> *%in, <32 x i32>* %out) #0 {
; CHECK-LABEL: subvector_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024: ld1w { [[DATA:z[0-9]+.s]] }, [[PG]]/z, [x0]
; VBITS_GE_1024: st1w { [[DATA]] }, [[PG]], [x1]
; CHECK: ret
  %a = load <32 x i32>, <32 x i32>* %in
  br label %bb1

bb1:
  store <32 x i32> %a, <32 x i32>* %out
  ret void
}

define void @subvector_v64i32(<64 x i32> *%in, <64 x i32>* %out) #0 {
; CHECK-LABEL: subvector_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048: ld1w { [[DATA:z[0-9]+.s]] }, [[PG]]/z, [x0]
; VBITS_GE_2048: st1w { [[DATA]] }, [[PG]], [x1]
; CHECK: ret
  %a = load <64 x i32>, <64 x i32>* %in
  br label %bb1

bb1:
  store <64 x i32> %a, <64 x i32>* %out
  ret void
}

;
define <8 x i1> @no_warn_dropped_scalable(<8 x i32>* %in) #0 {
; CHECK-LABEL: no_warn_dropped_scalable:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK: ld1w { [[A:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: cmpgt p{{[0-9]}}.s, [[PG]]/z, [[A]].s, z{{[0-9]+}}.s
; CHECK: ret
  %a = load <8 x i32>, <8 x i32>* %in
  br label %bb1

bb1:
  %cond = icmp sgt <8 x i32> %a, zeroinitializer
  ret <8 x i1> %cond
}

attributes #0 = { "target-features"="+sve" }
