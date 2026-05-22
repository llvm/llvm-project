; RUN: llc -mtriple=aarch64-linux-gnu -O3 -o - %s | FileCheck %s
;
; IR AFTER SimplifyCFG: the stores from both branches have been merged
; into a PHI + single store in a successor block. This separates the
; extractelement from the store into different basic blocks.
;
; ISel processes each basic block independently, so the extractelement
; in if.then has no store consumer in its block -- it lowers to UMOV (GPR).
; Late tail duplication sinks the store back into each branch, but the
; UMOV is already baked in, leaving a redundant cross-domain transfer:
;
;   umov  w8, v0.h[0]        ; FPR -> GPR (extractelement, lowered alone)
;   strh  w8, [x0]           ; GPR store  (sunk back by tail duplication)
;
; instead of the optimal:
;   str   h0, [x0]           ; direct FPR store
;
; The post-RA peephole in aarch64-ldst-opt fixes this by folding
; UMOVvi*_idx0 + GPR store --> FPR sub-register store.

; With the fix applied, if.then should use a direct FPR store.
; if.else legitimately needs the GPR (the XOR is a scalar GPR op),
; so the umov there is expected.

; CHECK-LABEL: extract_store_post_simplifycfg:
; CHECK:       // %bb.1: // %if.then
; CHECK-NOT:   umov
; CHECK:       str h{{[0-9]+}}, [x0]
; CHECK:       ret
; CHECK:       // %if.else
; CHECK:       umov
define void @extract_store_post_simplifycfg(ptr %p, <8 x i16> %vec, i32 %cond) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  call void asm sideeffect "", ""()
  %or = or <8 x i16> %vec, <i16 18, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  %lane.then = extractelement <8 x i16> %or, i64 0
  br label %if.end

if.else:
  %lane.else = extractelement <8 x i16> %vec, i64 0
  %xor = xor i16 %lane.else, 52
  br label %if.end

if.end:
  %storemerge = phi i16 [ %xor, %if.else ], [ %lane.then, %if.then ]
  store i16 %storemerge, ptr %p, align 2
  ret void
}
