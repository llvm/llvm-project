; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple

;CHECK-LABEL: Shuff:
;CHECK: tbl.8b
;CHECK: ret
define <8 x i8 > @Shuff(<8 x i8> %in, <8 x i8>* %out) nounwind ssp {
  %value = shufflevector <8 x i8> %in, <8 x i8> zeroinitializer, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i8> %value
}


