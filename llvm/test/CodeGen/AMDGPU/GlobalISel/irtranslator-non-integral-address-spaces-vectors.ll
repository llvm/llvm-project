; RUN: not --crash llc -global-isel -mtriple=amdgcn-amd-amdpal -mcpu=gfx900 -o - -stop-after=irtranslator < %s
; REQUIRES: asserts

; Confirm that no one's gotten vectors of addrspace(7) pointers to go through the
; IR translater incidentally.

define <2 x ptr addrspace(7)> @no_auto_constfold_gep_vector() {
  %gep = getelementptr i8, <2 x ptr addrspace(7)> zeroinitializer, <2 x i32> <i32 123, i32 123>
  ret <2 x ptr addrspace(7)> %gep
}

define <2 x ptr addrspace(7)> @gep_vector_splat(<2 x ptr addrspace(7)> %ptrs, i64 %idx) {
  %gep = getelementptr i8, <2 x ptr addrspace(7)> %ptrs, i64 %idx
  ret <2 x ptr addrspace(7)> %gep
}
