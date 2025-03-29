; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
; llvm.org/PR25439
; The dominance of the generated non-affine subregion block was based on the
; scop's merge block, therefore resulted in an invalid DominanceTree.
; It resulted in some values as assumed to be unusable in the actual generated
; exit block. Here we check whether the value %escaping is taken from the
; generated block.
;
; CHECK-LABEL: polly.stmt.subregion_entry:
; CHECK:         %p_escaping = select i1 undef, i32 undef, i32 undef
;
; CHECK-LABEL: polly.stmt.polly.merge_new_and_old.exit:
; CHECK:         store i32 %p_escaping, ptr %escaping.s2a

define i32 @func() {
entry:
  br label %subregion_entry

subregion_entry:
  %escaping = select i1 undef, i32 undef, i32 undef
  %cond = or i1 undef, undef
  br i1 %cond, label %subregion_exit, label %subregion_if

subregion_if:
  br label %subregion_exit

subregion_exit:
  ret i32 %escaping
}
