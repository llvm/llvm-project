; RUN: split-file %s %t
; RUN: not llc -mtriple=amdgpu12.50 -filetype=null %t/mark.ll 2>&1 | FileCheck %s -check-prefix=MARK
; RUN: not llc -mtriple=amdgpu12.50 -filetype=null %t/wait.ll 2>&1 | FileCheck %s -check-prefix=WAIT

; The stage mask operand carries a Range covering the bits of the defined
; stages, so the IR verifier rejects a mask naming a stage that does not exist
; before instruction selection. Bits of reserved stages are inside the range and
; so are accepted: omitting a stage whose operations do not exist yet is
; harmless, and keeps masks portable as the reserved slots are filled in.

;--- mark.ll
; 2048 is the bit just past the last stage, and 4096 is further out still.
; MARK: immarg value 2048 for arg 0 out of range [0,2048)
; MARK: immarg value 4096 for arg 0 out of range [0,2048)
define amdgpu_kernel void @asyncmark_mask_out_of_range() {
  call void @llvm.amdgcn.asyncmark(i32 2048)
  call void @llvm.amdgcn.asyncmark(i32 4096)
  ret void
}

;--- wait.ll
; A mask made only of reserved stage bits is fine, so the sole error here is the
; out-of-range one.
; WAIT: immarg value 2048 for arg 1 out of range [0,2048)
; WAIT-NOT: out of range
define amdgpu_kernel void @wait_asyncmark_mask_out_of_range() {
  call void @llvm.amdgcn.wait.asyncmark(i16 0, i32 2000)
  call void @llvm.amdgcn.wait.asyncmark(i16 0, i32 2048)
  ret void
}
