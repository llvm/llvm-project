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

define <8 x i1> @no_warn_dropped_scalable(<8 x i32>* %in) #0 {
; CHECK-LABEL: no_warn_dropped_scalable:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK: ld1w { [[A:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK: cmpgt p{{[0-9]}}.s, [[PG]]/z, [[A]].s, #0
; CHECK: ret
  %a = load <8 x i32>, <8 x i32>* %in
  br label %bb1

bb1:
  %cond = icmp sgt <8 x i32> %a, zeroinitializer
  ret <8 x i1> %cond
}

; binop(insert_subvec(a), insert_subvec(b)) -> insert_subvec(binop(a,b)) like
; combines remove redundant subvector operations. This test ensures it's not
; performed when the input idiom is the result of operation legalisation. When
; not prevented the test triggers infinite combine->legalise->combine->...
define void @no_subvector_binop_hang(<8 x i32>* %in, <8 x i32>* %out, i1 %cond) #0 {
; CHECK-LABEL: no_subvector_binop_hang:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT:    ld1w { [[A:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT:    ld1w { [[B:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT:    tbz w2, #0, .LBB5_2
; CHECK-NEXT:  // %bb.1: // %bb.1
; CHECK-NEXT:    orr [[OR:z[0-9]+]].d, [[A]].d, [[B]].d
; CHECK-NEXT:    st1w { [[OR]].s }, [[PG]], [x1]
; CHECK-NEXT:  .LBB5_2: // %bb.2
; CHECK-NEXT:    ret
  %a = load <8 x i32>, <8 x i32>* %in
  %b = load <8 x i32>, <8 x i32>* %out
  br i1 %cond, label %bb.1, label %bb.2

bb.1:
  %or = or <8 x i32> %a, %b
  store <8 x i32> %or, <8 x i32>* %out
  br label %bb.2

bb.2:
  ret void
}

attributes #0 = { "target-features"="+sve" }
