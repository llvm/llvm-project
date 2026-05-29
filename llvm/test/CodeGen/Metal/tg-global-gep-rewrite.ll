; MetalPrepare inserts a base GEP (__base_<name>) at the entry of any function
; that touches an addrspace(3) array global, so downstream byte-offset GEPs
; resolve against the typed base rather than the global directly. The test
; documents the current shape; the post-pass IR keeps the original i8 array
; type and uses an i8 element-stride GEP to index it.
;
; TODO(metal-revisit GH-XXX): the eventual goal is to retype the global into
; the inferred element type (e.g. [32 x float]) and emit float-stride GEPs;
; the rewrite here is a documented intermediate state.
;
; RUN: llc -mtriple=air -stop-after=metal-prepare %s -o - | FileCheck %s

@tg = internal addrspace(3) global [128 x i8] undef, align 4

; CHECK: @tg = internal addrspace(3) global [128 x i8]

; CHECK-LABEL: define void @store_float_to_tg
; CHECK: %__base_tg = getelementptr inbounds [128 x i8], ptr addrspace(3) @tg
; CHECK: getelementptr inbounds i8, ptr addrspace(3) %__base_tg
; CHECK: store float
define void @store_float_to_tg(float %v, i64 %byteIdx) {
entry:
  %p = getelementptr inbounds i8, ptr addrspace(3) @tg, i64 %byteIdx
  store float %v, ptr addrspace(3) %p, align 4
  ret void
}
