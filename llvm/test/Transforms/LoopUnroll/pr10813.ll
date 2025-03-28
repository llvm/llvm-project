; RUN: opt < %s -passes=loop-unroll -disable-output

define void @"f_fu___REFUf[]REFUf[]Uf"(i1 %arg) nounwind {
allocas:
  br i1 %arg, label %cif_mask_all, label %cif_mixed_test_all

cif_mask_all:                                     ; preds = %allocas
  unreachable

cif_mixed_test_all:                               ; preds = %allocas
  br label %pl_loop.i964

pl_loop.i964:                                     ; preds = %pl_loopend.i973, %cif_mixed_test_all
  %0 = phi i32 [ %pl_nextlane.i971, %pl_loopend.i973 ], [ 0, %cif_mixed_test_all ]
  br i1 %arg, label %pl_dolane.i970, label %pl_loopend.i973

pl_dolane.i970:                                   ; preds = %pl_loop.i964
  %storeval.i.i969 = extractelement <4 x i8> <i8 0, i8 1, i8 2, i8 3>, i32 %0
  store i8 %storeval.i.i969, ptr undef, align 1
  br label %pl_loopend.i973

pl_loopend.i973:                                  ; preds = %pl_dolane.i970, %pl_loop.i964
  %pl_nextlane.i971 = add i32 %0, 1
  %exitcond5 = icmp ne i32 %pl_nextlane.i971, 5
  br i1 %exitcond5, label %pl_loop.i964, label %__scatter_base_offsets_i8.exit974

__scatter_base_offsets_i8.exit974:                ; preds = %pl_loopend.i973
  unreachable
}
