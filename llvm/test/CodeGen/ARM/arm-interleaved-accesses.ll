; RUN: llc -mtriple=arm-eabi -mattr=+neon -lower-interleaved-accesses=true < %s | FileCheck %s -check-prefix=NEON
; RUN: llc -mtriple=arm-eabi -mattr=-neon -lower-interleaved-accesses=true < %s | FileCheck %s -check-prefix=NONEON

; NEON-LABEL: load_factor2:
; NEON: vld2.8 {d16, d17}, [r0]
; NONEON-LABEL: load_factor2:
; NONEON-NOT: vld2
define <8 x i8> @load_factor2(<16 x i8>* %ptr) {
  %wide.vec = load <16 x i8>, <16 x i8>* %ptr, align 4
  %strided.v0 = shufflevector <16 x i8> %wide.vec, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %strided.v1 = shufflevector <16 x i8> %wide.vec, <16 x i8> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %add = add nsw <8 x i8> %strided.v0, %strided.v1
  ret <8 x i8> %add
}

; NEON-LABEL: load_factor3:
; NEON: vld3.32 {d16, d17, d18}, [r0]
; NONEON-LABEL: load_factor3:
; NONEON-NOT: vld3
define <2 x i32> @load_factor3(i32* %ptr) {
  %base = bitcast i32* %ptr to <6 x i32>*
  %wide.vec = load <6 x i32>, <6 x i32>* %base, align 4
  %strided.v2 = shufflevector <6 x i32> %wide.vec, <6 x i32> undef, <2 x i32> <i32 2, i32 5>
  %strided.v1 = shufflevector <6 x i32> %wide.vec, <6 x i32> undef, <2 x i32> <i32 1, i32 4>
  %add = add nsw <2 x i32> %strided.v2, %strided.v1
  ret <2 x i32> %add
}

; NEON-LABEL: load_factor4:
; NEON: vld4.32 {d16, d18, d20, d22}, [r0]!
; NEON: vld4.32 {d17, d19, d21, d23}, [r0]
; NONEON-LABEL: load_factor4:
; NONEON-NOT: vld4
define <4 x i32> @load_factor4(i32* %ptr) {
  %base = bitcast i32* %ptr to <16 x i32>*
  %wide.vec = load <16 x i32>, <16 x i32>* %base, align 4
  %strided.v0 = shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %strided.v2 = shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %add = add nsw <4 x i32> %strided.v0, %strided.v2
  ret <4 x i32> %add
}

; NEON-LABEL: store_factor2:
; NEON: vst2.8 {d16, d17}, [r0]
; NONEON-LABEL: store_factor2:
; NONEON-NOT: vst2
define void @store_factor2(<16 x i8>* %ptr, <8 x i8> %v0, <8 x i8> %v1) {
  %interleaved.vec = shufflevector <8 x i8> %v0, <8 x i8> %v1, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  store <16 x i8> %interleaved.vec, <16 x i8>* %ptr, align 4
  ret void
}

; NEON-LABEL: store_factor3:
; NEON: vst3.32 {d16, d18, d20}, [r0]!
; NEON: vst3.32 {d17, d19, d21}, [r0]
; NONEON-LABEL: store_factor3:
; NONEON-NOT: vst3.32
define void @store_factor3(i32* %ptr, <4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2) {
  %base = bitcast i32* %ptr to <12 x i32>*
  %v0_v1 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v2_u = shufflevector <4 x i32> %v2, <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <8 x i32> %v0_v1, <8 x i32> %v2_u, <12 x i32> <i32 0, i32 4, i32 8, i32 1, i32 5, i32 9, i32 2, i32 6, i32 10, i32 3, i32 7, i32 11>
  store <12 x i32> %interleaved.vec, <12 x i32>* %base, align 4
  ret void
}

; NEON-LABEL: store_factor4:
; NEON: vst4.32 {d16, d18, d20, d22}, [r0]!
; NEON: vst4.32 {d17, d19, d21, d23}, [r0]
; NONEON-LABEL: store_factor4:
; NONEON-NOT: vst4
define void @store_factor4(i32* %ptr, <4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3) {
  %base = bitcast i32* %ptr to <16 x i32>*
  %v0_v1 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v2_v3 = shufflevector <4 x i32> %v2, <4 x i32> %v3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec = shufflevector <8 x i32> %v0_v1, <8 x i32> %v2_v3, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x i32> %interleaved.vec, <16 x i32>* %base, align 4
  ret void
}

; The following cases test that interleaved access of pointer vectors can be
; matched to ldN/stN instruction.

; NEON-LABEL: load_ptrvec_factor2:
; NEON: vld2.32 {d16, d17}, [r0]
; NONEON-LABEL: load_ptrvec_factor2:
; NONEON-NOT: vld2
define <2 x i32*> @load_ptrvec_factor2(i32** %ptr) {
  %base = bitcast i32** %ptr to <4 x i32*>*
  %wide.vec = load <4 x i32*>, <4 x i32*>* %base, align 4
  %strided.v0 = shufflevector <4 x i32*> %wide.vec, <4 x i32*> undef, <2 x i32> <i32 0, i32 2>
  ret <2 x i32*> %strided.v0
}

; NEON-LABEL: load_ptrvec_factor3:
; NEON: vld3.32 {d16, d17, d18}, [r0]
; NONEON-LABEL: load_ptrvec_factor3:
; NONEON-NOT: vld3
define void @load_ptrvec_factor3(i32** %ptr, <2 x i32*>* %ptr1, <2 x i32*>* %ptr2) {
  %base = bitcast i32** %ptr to <6 x i32*>*
  %wide.vec = load <6 x i32*>, <6 x i32*>* %base, align 4
  %strided.v2 = shufflevector <6 x i32*> %wide.vec, <6 x i32*> undef, <2 x i32> <i32 2, i32 5>
  store <2 x i32*> %strided.v2, <2 x i32*>* %ptr1
  %strided.v1 = shufflevector <6 x i32*> %wide.vec, <6 x i32*> undef, <2 x i32> <i32 1, i32 4>
  store <2 x i32*> %strided.v1, <2 x i32*>* %ptr2
  ret void
}

; NEON-LABEL: load_ptrvec_factor4:
; NEON: vld4.32 {d16, d17, d18, d19}, [r0]
; NONEON-LABEL: load_ptrvec_factor4:
; NONEON-NOT: vld4
define void @load_ptrvec_factor4(i32** %ptr, <2 x i32*>* %ptr1, <2 x i32*>* %ptr2) {
  %base = bitcast i32** %ptr to <8 x i32*>*
  %wide.vec = load <8 x i32*>, <8 x i32*>* %base, align 4
  %strided.v1 = shufflevector <8 x i32*> %wide.vec, <8 x i32*> undef, <2 x i32> <i32 1, i32 5>
  %strided.v3 = shufflevector <8 x i32*> %wide.vec, <8 x i32*> undef, <2 x i32> <i32 3, i32 7>
  store <2 x i32*> %strided.v1, <2 x i32*>* %ptr1
  store <2 x i32*> %strided.v3, <2 x i32*>* %ptr2
  ret void
}

; NEON-LABEL: store_ptrvec_factor2:
; NEON: vst2.32 {d16, d17}, [r0]
; NONEON-LABEL: store_ptrvec_factor2:
; NONEON-NOT: vst2
define void @store_ptrvec_factor2(i32** %ptr, <2 x i32*> %v0, <2 x i32*> %v1) {
  %base = bitcast i32** %ptr to <4 x i32*>*
  %interleaved.vec = shufflevector <2 x i32*> %v0, <2 x i32*> %v1, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x i32*> %interleaved.vec, <4 x i32*>* %base, align 4
  ret void
}

; NEON-LABEL: store_ptrvec_factor3:
; NEON: vst3.32 {d16, d17, d18}, [r0]
; NONEON-LABEL: store_ptrvec_factor3:
; NONEON-NOT: vst3
define void @store_ptrvec_factor3(i32** %ptr, <2 x i32*> %v0, <2 x i32*> %v1, <2 x i32*> %v2) {
  %base = bitcast i32** %ptr to <6 x i32*>*
  %v0_v1 = shufflevector <2 x i32*> %v0, <2 x i32*> %v1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v2_u = shufflevector <2 x i32*> %v2, <2 x i32*> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <4 x i32*> %v0_v1, <4 x i32*> %v2_u, <6 x i32> <i32 0, i32 2, i32 4, i32 1, i32 3, i32 5>
  store <6 x i32*> %interleaved.vec, <6 x i32*>* %base, align 4
  ret void
}

; NEON-LABEL: store_ptrvec_factor4:
; NEON: vst4.32 {d16, d17, d18, d19}, [r0]
; NONEON-LABEL: store_ptrvec_factor4:
; NONEON-NOT: vst4
define void @store_ptrvec_factor4(i32* %ptr, <2 x i32*> %v0, <2 x i32*> %v1, <2 x i32*> %v2, <2 x i32*> %v3) {
  %base = bitcast i32* %ptr to <8 x i32*>*
  %v0_v1 = shufflevector <2 x i32*> %v0, <2 x i32*> %v1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v2_v3 = shufflevector <2 x i32*> %v2, <2 x i32*> %v3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %interleaved.vec = shufflevector <4 x i32*> %v0_v1, <4 x i32*> %v2_v3, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>
  store <8 x i32*> %interleaved.vec, <8 x i32*>* %base, align 4
  ret void
}

; Following cases check that shuffle maskes with undef indices can be matched
; into ldN/stN instruction.

; NEON-LABEL: load_undef_mask_factor2:
; NEON: vld2.32 {d16, d17, d18, d19}, [r0]
; NONEON-LABEL: load_undef_mask_factor2:
; NONEON-NOT: vld2
define <4 x i32> @load_undef_mask_factor2(i32* %ptr) {
  %base = bitcast i32* %ptr to <8 x i32>*
  %wide.vec = load <8 x i32>, <8 x i32>* %base, align 4
  %strided.v0 = shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 undef, i32 2, i32 undef, i32 6>
  %strided.v1 = shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 undef, i32 3, i32 undef, i32 7>
  %add = add nsw <4 x i32> %strided.v0, %strided.v1
  ret <4 x i32> %add
}

; NEON-LABEL: load_undef_mask_factor3:
; NEON: vld3.32 {d16, d18, d20}, [r0]!
; NEON: vld3.32 {d17, d19, d21}, [r0]
; NONEON-LABEL: load_undef_mask_factor3:
; NONEON-NOT: vld3
define <4 x i32> @load_undef_mask_factor3(i32* %ptr) {
  %base = bitcast i32* %ptr to <12 x i32>*
  %wide.vec = load <12 x i32>, <12 x i32>* %base, align 4
  %strided.v2 = shufflevector <12 x i32> %wide.vec, <12 x i32> undef, <4 x i32> <i32 2, i32 undef, i32 undef, i32 undef>
  %strided.v1 = shufflevector <12 x i32> %wide.vec, <12 x i32> undef, <4 x i32> <i32 1, i32 4, i32 7, i32 10>
  %add = add nsw <4 x i32> %strided.v2, %strided.v1
  ret <4 x i32> %add
}

; NEON-LABEL: load_undef_mask_factor4:
; NEON: vld4.32 {d16, d18, d20, d22}, [r0]!
; NEON: vld4.32 {d17, d19, d21, d23}, [r0]
; NONEON-LABEL: load_undef_mask_factor4:
; NONEON-NOT: vld4
define <4 x i32> @load_undef_mask_factor4(i32* %ptr) {
  %base = bitcast i32* %ptr to <16 x i32>*
  %wide.vec = load <16 x i32>, <16 x i32>* %base, align 4
  %strided.v0 = shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <4 x i32> <i32 0, i32 4, i32 undef, i32 undef>
  %strided.v2 = shufflevector <16 x i32> %wide.vec, <16 x i32> undef, <4 x i32> <i32 2, i32 6, i32 undef, i32 undef>
  %add = add nsw <4 x i32> %strided.v0, %strided.v2
  ret <4 x i32> %add
}

; NEON-LABEL: store_undef_mask_factor2:
; NEON: vst2.32 {d16, d17, d18, d19}, [r0]
; NONEON-LABEL: store_undef_mask_factor2:
; NONEON-NOT: vst2
define void @store_undef_mask_factor2(i32* %ptr, <4 x i32> %v0, <4 x i32> %v1) {
  %base = bitcast i32* %ptr to <8 x i32>*
  %interleaved.vec = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 2, i32 6, i32 3, i32 7>
  store <8 x i32> %interleaved.vec, <8 x i32>* %base, align 4
  ret void
}

; NEON-LABEL: store_undef_mask_factor3:
; NEON: vst3.32 {d16, d18, d20}, [r0]!
; NEON: vst3.32 {d17, d19, d21}, [r0]
; NONEON-LABEL: store_undef_mask_factor3:
; NONEON-NOT: vst3
define void @store_undef_mask_factor3(i32* %ptr, <4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2) {
  %base = bitcast i32* %ptr to <12 x i32>*
  %v0_v1 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v2_u = shufflevector <4 x i32> %v2, <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <8 x i32> %v0_v1, <8 x i32> %v2_u, <12 x i32> <i32 0, i32 4, i32 undef, i32 1, i32 undef, i32 9, i32 2, i32 6, i32 10, i32 3, i32 7, i32 11>
  store <12 x i32> %interleaved.vec, <12 x i32>* %base, align 4
  ret void
}

; NEON-LABEL: store_undef_mask_factor4:
; NEON: vst4.32 {d16, d18, d20, d22}, [r0]!
; NEON: vst4.32 {d17, d19, d21, d23}, [r0]
; NONEON-LABEL: store_undef_mask_factor4:
; NONEON-NOT: vst4
define void @store_undef_mask_factor4(i32* %ptr, <4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3) {
  %base = bitcast i32* %ptr to <16 x i32>*
  %v0_v1 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v2_v3 = shufflevector <4 x i32> %v2, <4 x i32> %v3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec = shufflevector <8 x i32> %v0_v1, <8 x i32> %v2_v3, <16 x i32> <i32 0, i32 4, i32 8, i32 undef, i32 undef, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x i32> %interleaved.vec, <16 x i32>* %base, align 4
  ret void
}

; The following test cases check that address spaces are properly handled

; NEON-LABEL: load_address_space
; NEON: vld3.32
; NONEON-LABEL: load_address_space
; NONEON-NOT: vld3
define void @load_address_space(<4 x i32> addrspace(1)* %A, <2 x i32>* %B) {
 %tmp = load <4 x i32>, <4 x i32> addrspace(1)* %A
 %interleaved = shufflevector <4 x i32> %tmp, <4 x i32> undef, <2 x i32> <i32 0, i32 3>
 store <2 x i32> %interleaved, <2 x i32>* %B
 ret void
}

; NEON-LABEL: store_address_space
; NEON: vst2.32
; NONEON-LABEL: store_address_space
; NONEON-NOT: vst2
define void @store_address_space(<2 x i32>* %A, <2 x i32>* %B, <4 x i32> addrspace(1)* %C) {
 %tmp0 = load <2 x i32>, <2 x i32>* %A
 %tmp1 = load <2 x i32>, <2 x i32>* %B
 %interleaved = shufflevector <2 x i32> %tmp0, <2 x i32> %tmp1, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
 store <4 x i32> %interleaved, <4 x i32> addrspace(1)* %C
 ret void
}

; Check that we do something sane with illegal types.

; NEON-LABEL: load_illegal_factor2:
; NEON: BB#0:
; NEON-NEXT: vld1.64 {d16, d17}, [r0:128]
; NEON-NEXT: vuzp.32 q8, {{.*}}
; NEON-NEXT: vmov r0, r1, d16
; NEON-NEXT: vmov r2, r3, {{.*}}
; NEON-NEXT: mov pc, lr
; NONEON-LABEL: load_illegal_factor2:
; NONEON: BB#0:
; NONEON-NEXT: ldr [[ELT0:r[0-9]+]], [r0]
; NONEON-NEXT: ldr r1, [r0, #8]
; NONEON-NEXT: mov r0, [[ELT0]]
; NONEON-NEXT: mov pc, lr
define <3 x float> @load_illegal_factor2(<3 x float>* %p) nounwind {
  %tmp1 = load <3 x float>, <3 x float>* %p, align 16
  %tmp2 = shufflevector <3 x float> %tmp1, <3 x float> undef, <3 x i32> <i32 0, i32 2, i32 undef>
  ret <3 x float> %tmp2
}

; This lowering isn't great, but it's at least correct.

; NEON-LABEL: store_illegal_factor2:
; NEON: BB#0:
; NEON-NEXT: vldr d17, [sp]
; NEON-NEXT: vmov d16, r2, r3
; NEON-NEXT: vuzp.32 q8, {{.*}}
; NEON-NEXT: vstr d16, [r0]
; NEON-NEXT: mov pc, lr
; NONEON-LABEL: store_illegal_factor2:
; NONEON: BB#0:
; NONEON-NEXT: stm r0, {r1, r3}
; NONEON-NEXT: mov pc, lr
define void @store_illegal_factor2(<3 x float>* %p, <3 x float> %v) nounwind {
  %tmp1 = shufflevector <3 x float> %v, <3 x float> undef, <3 x i32> <i32 0, i32 2, i32 undef>
  store <3 x float> %tmp1, <3 x float>* %p, align 16
  ret void
}

; NEON-LABEL: load_factor2_with_extract_user:
; NEON: vld2.32 {d16, d17, d18, d19}, [r0:64]
; NEON: vmov.32 r0, d16[1]
; NONEON-LABEL: load_factor2_with_extract_user:
; NONEON-NOT: vld2
define i32 @load_factor2_with_extract_user(<8 x i32>* %a) {
  %1 = load <8 x i32>, <8 x i32>* %a, align 8
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %3 = extractelement <8 x i32> %1, i32 2
  ret i32 %3
}
