; RUN: opt -S -passes=instcombine < %s

; This regression test is verifying that the optimization defined by
; canReplaceGEPIdxWithZero, which replaces a GEP index with zero iff we can show
; a value other than zero would cause undefined behaviour, does not throw a
; 'assumption that TypeSize is not scalable' warning when the source element type
; is a scalable vector.

; If the source element is a scalable vector type, then we cannot deduce whether
; or not indexing at a given index is undefined behaviour, because the size of
; the vector is not known.

declare void @do_something(<vscale x 4 x i32> %x)

define void @can_replace_gep_idx_with_zero_typesize(i64 %n, ptr %a, i64 %b) {
  %idx = getelementptr <vscale x 4 x i32>, ptr %a, i64 %b
  %tmp = load <vscale x 4 x i32>, ptr %idx
  call void @do_something(<vscale x 4 x i32> %tmp)
  ret void
}

define void @can_replace_gep_idx_with_zero_typesize_2(i64 %n, ptr %a, i64 %b) {
  %idx = getelementptr [2 x <vscale x 4 x i32>], ptr %a, i64 %b, i64 0
  %tmp = load <vscale x 4 x i32>, ptr %idx
  call void @do_something(<vscale x 4 x i32> %tmp)
  ret void
}
