; RUN: opt -S -instcombine < %s 2>%t
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; This regression test is verifying that the optimization defined by
; canReplaceGEPIdxWithZero, which replaces a GEP index with zero iff we can show
; a value other than zero would cause undefined behaviour, does not throw a
; 'assumption that TypeSize is not scalable' warning when the source element type
; is a scalable vector.

; If the source element is a scalable vector type, then we cannot deduce whether
; or not indexing at a given index is undefined behaviour, because the size of
; the vector is not known.

; If this check fails please read test/CodeGen/AArch64/README for instructions
; on how to resolve it.
; WARN-NOT: warning: {{.*}}TypeSize is not scalable

declare void @do_something(<vscale x 4 x i32> %x)

define void @can_replace_gep_idx_with_zero_typesize(i64 %n, <vscale x 4 x i32>* %a, i64 %b) {
  %idx = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %a, i64 %b
  %tmp = load <vscale x 4 x i32>, <vscale x 4 x i32>* %idx
  call void @do_something(<vscale x 4 x i32> %tmp)
  ret void
}
